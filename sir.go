package dimred

import (
	"fmt"
	"log"
	"math"
	"os"
	"sort"

	"github.com/gonum/blas"
	"github.com/gonum/blas/blas64"
	"github.com/gonum/floats"
	"github.com/gonum/matrix/mat64"
	"github.com/kshedden/dstream/dstream"
)

// SliceFunc returns the slice index for a data record
type SliceFunc func(float64) int

type SIR struct {

	// The data used to perform the analysis
	Data dstream.Reg

	// A function returning the slice index of each response value
	Slicer SliceFunc

	// The number of directions to retain.  If 0, all directions
	// are retained.
	NDir int

	// The estimated directions
	Dir [][]float64

	// The eigenvalues
	Eig []float64

	ns []int // Sample size per slice
	nc []int // Sample size per chunk

	smn [][]float64 // Slice means
	cmn [][]float64 // Chunk means

	// Raw float slices containing moments
	ccvf  [][]float64 // Chunk covariances
	mcovb []float64   // Marginal covariance
	ccmnb []float64   // Covariance of conditional means

	// Matrix objects containing moments
	mcov mat64.Symmetric // Marginal covariance
	ccmn mat64.Symmetric // Covariance of conditional means

	// Used for covariance projection
	projBasis  mat64.Matrix
	ccmnNoProj mat64.Symmetric
	mcovNoProj mat64.Symmetric

	// Optional log
	log *log.Logger

	mn       []float64 // Marginal mean
	n        int       // Overall sample size
	nchunk   int
	nskip    int  // Number of records skip since slice index was negative
	doneInit bool // true if Init has run
}

func (sir *SIR) SetLogFile(filename string) {

	fid, err := os.Create(filename)
	if err != nil {
		panic(err)
	}

	sir.log = log.New(fid, "", log.Lshortfile)
}

// reflect copies the lower triangle of a square matrix into the upper
// triangle
func reflect(x []float64, p int) {
	for j1 := 0; j1 < p; j1++ {
		for j2 := 0; j2 < j1; j2++ {
			x[j2*p+j1] = x[j1*p+j2]
		}
	}
}

// getstats walks through the data and calculates various summary
// statistics.
func (sir *SIR) getstats() {

	sir.Data.Reset()
	p := sir.Data.NumCov()
	pp := p * p

	var sl []int // slice indices for current chunk

	for sir.Data.Next() {

		y := sir.Data.YData()
		sir.n += len(y)
		sir.nchunk++
		sir.nc = append(sir.nc, len(y))

		// Mean of current chunk
		chm := make([]float64, p)
		for j := 0; j < p; j++ {
			x := sir.Data.XData(j)
			chm[j] = floats.Sum(x) / float64(len(x))
		}
		sir.cmn = append(sir.cmn, chm)

		// Get the slice values and update the slice counts
		sl = resizeInt(sl, len(y))
		for i, v := range y {
			sl[i] = sir.Slicer(v)
			if sl[i] < 0 {
				sir.nskip++
				continue
			}
			sir.n++

			// Grow if needed
			for len(sir.ns) <= sl[i] {
				sir.ns = append(sir.ns, 0)
				sir.smn = append(sir.smn, make([]float64, p))
			}
		}

		cv := make([]float64, pp) // within-chunk covariance
		for j1 := 0; j1 < p; j1++ {
			x1 := sir.Data.XData(j1)

			// Update the slice mean and count
			for i, s := range sl {
				if s < 0 {
					continue
				}
				sir.smn[s][j1] += x1[i]
				if j1 == 0 {
					sir.ns[s]++
				}
			}

			// Update the within-chunk covariance
			for j2 := 0; j2 <= j1; j2++ {
				x2 := sir.Data.XData(j2)
				for i, s := range sl {
					if s >= 0 {
						cv[j1*p+j2] += (x1[i] - chm[j1]) * (x2[i] - chm[j2])
					}
				}
			}
		}

		reflect(cv, p)
		floats.Scale(1/float64(len(y)), cv)
		sir.ccvf = append(sir.ccvf, cv)
	}

	// Normalize slice means
	for j, n := range sir.ns {
		floats.Scale(1/float64(n), sir.smn[j])
	}
}

func (sir *SIR) SliceSizes() []int {
	return sir.ns
}

func (sir *SIR) SliceMeans() [][]float64 {
	return sir.smn
}

// marg pools over the chunks to obtain the marginal mean and covariance
func (sir *SIR) marg() {

	p := sir.Data.NumCov()
	pp := p * p

	// Get the marginal mean
	mn := make([]float64, p)
	var w float64
	for i, mu := range sir.cmn {
		wt := float64(sir.nc[i])
		w += wt
		floats.AddScaled(mn, wt, mu)
	}
	floats.Scale(1/w, mn)
	sir.mn = mn

	// Get the within and between chunk covariances
	w = 0
	bc := make([]float64, pp)
	wc := make([]float64, pp)
	for i, mu := range sir.cmn {

		wt := float64(sir.nc[i])
		w += wt
		for j1 := 0; j1 < p; j1++ {
			for j2 := 0; j2 <= j1; j2++ {
				u := (mu[j1] - mn[j1]) * (mu[j2] - mn[j2])
				bc[j1*p+j2] += wt * u
			}
		}

		floats.AddScaled(wc, wt, sir.ccvf[i])
	}

	reflect(bc, p)
	floats.Scale(1/float64(w), bc)
	floats.Scale(1/float64(w), wc)

	sir.mcovb = make([]float64, pp)
	floats.AddTo(sir.mcovb, wc, bc)

	sir.mcov = mat64.NewSymDense(p, sir.mcovb)
}

// getccmn calculates Cov E[X|Y], the covariance of the conditional
// means.
func (sir *SIR) getccmn() {

	p := sir.Data.NumCov()
	pp := p * p
	mn := sir.mn
	ccmn := make([]float64, pp)

	var w float64 // total weight
	for j, sm := range sir.smn {
		wt := float64(sir.ns[j])
		if wt <= 0 {
			continue
		}
		w += wt
		for i1 := 0; i1 < p; i1++ {
			for i2 := 0; i2 <= i1; i2++ {
				u := (sm[i1] - mn[i1]) * (sm[i2] - mn[i2])
				ccmn[i1*p+i2] += wt * u
			}
		}
	}

	// Normalize and reflect
	for j1 := 0; j1 < p; j1++ {
		for j2 := 0; j2 <= j1; j2++ {
			ccmn[j1*p+j2] /= w
			ccmn[j2*p+j1] = ccmn[j1*p+j2]
		}
	}

	sir.ccmnb = ccmn
	sir.ccmn = mat64.NewSymDense(p, ccmn)
}

// Init calculates summary statistics but does not calculate the EDR
// directions.  Fit calls Init automatically if needed, but Init
// should be called explicitly if prior to calling ProjectEigen or
// other modifiers prior to fit.
func (sir *SIR) Init() {
	sir.getstats()
	sir.marg()
	sir.getccmn()
	sir.doneInit = true
	if sir.log != nil {
		sir.log.Printf("%d variables\n", sir.Data.NumCov())
		sir.log.Printf("%d data records used\n", sir.n)
		sir.log.Printf("%d chunks read\n", sir.nchunk)
		sir.log.Printf("%d data records with negative slice index skipped\n", sir.nskip)

		sir.log.Printf("Slice sample sizes:\n")
		for j, n := range sir.ns {
			sir.log.Printf("%6d    %6d\n", j, n)
		}
	}
}

func (sir *SIR) MargCov() mat64.Symmetric {
	if sir.mcovNoProj != nil {
		return sir.mcovNoProj
	}
	return sir.mcov
}

func (sir *SIR) CovMean() mat64.Symmetric {
	if sir.ccmnNoProj != nil {
		return sir.ccmnNoProj
	}
	return sir.ccmn
}

func (sir *SIR) MargCovEigs() []float64 {

	var es mat64.EigenSym
	ok := es.Factorize(sir.mcov, false)
	if !ok {
		panic("unable to determine eigenvectors of marginal covariance")
	}

	ev := es.Values(nil)
	sort.Sort(sort.Reverse(sort.Float64Slice(ev)))
	return ev
}

// conjugate returns b'* a * b, a must be non-singular
func conjugate(a mat64.Symmetric, b mat64.Matrix) mat64.Symmetric {
	q1 := new(mat64.Dense)
	q1.Mul(b.T(), a)
	q2 := new(mat64.Dense)
	q2.Mul(b.T(), q1.T())
	ma := q2.RawMatrix()
	sy := new(blas64.Symmetric)
	sy.N = ma.Rows
	sy.Stride = ma.Stride
	sy.Data = ma.Data
	sy.Uplo = blas.Upper
	q3 := new(mat64.SymDense)
	q3.SetRawSymmetric(*sy)
	return q3
}

// ProjectEigen projects the marginal and conditional covariance
// matrices to the space spanned by a given number of eigenvectors of
// the marginal covariance matrix.
func (sir *SIR) ProjectEigen(ndim int) {

	p := sir.Data.NumCov()
	mcov := sir.mcov

	es := new(mat64.EigenSym)
	ok := es.Factorize(mcov, true)
	if !ok {
		panic("unable to determine eigenvectors of marginal covariance")
	}

	evec := new(mat64.Dense)
	evec.EigenvectorsSym(es)
	if sir.log != nil {
		sir.log.Printf("Eigenvalues of marginal covariance:")
		sir.log.Printf(fmt.Sprintf("%v\n", es.Values(nil)))
		sir.log.Printf(fmt.Sprintf("Retaining %d-dimensional eigenspace, dropping %d dimensions\n", ndim, p-ndim))
	}
	evecv := evec.View(0, p-ndim, p, ndim)

	sir.projBasis = evecv
	sir.mcovNoProj = sir.mcov
	sir.mcov = conjugate(sir.mcov, evecv)
	sir.ccmnNoProj = sir.ccmn
	sir.ccmn = conjugate(sir.ccmn, evecv)
}

func (sir *SIR) Fit() {

	if !sir.doneInit {
		sir.Init()
	}

	p := sir.Data.NumCov()
	mcov := sir.mcov
	ccmn := sir.ccmn

	// Number of directions to retain
	ndir := sir.NDir
	if ndir == 0 {
		ndir = p
	}

	msr := new(mat64.Cholesky)
	if ok := msr.Factorize(mcov); !ok {
		// Provide error information before exiting
		print("Marginal covariance is not PSD\n")
		ei := new(mat64.EigenSym)
		if ok := ei.Factorize(mcov, false); !ok {
			panic("Cannot obtain eigenvalues either.\n")
		}
		os.Stderr.Write([]byte("The eigenvalues are:\n"))
		os.Stderr.Write([]byte(fmt.Sprintf("%v\n", ei.Values(nil))))
	}

	q1 := new(mat64.TriDense)
	q2 := new(mat64.Dense)
	q3 := new(mat64.Dense)
	q1.LFromCholesky(msr)

	q2.Solve(q1, ccmn)
	q3.Solve(q1, q2.T())

	// q3 is a symmetric matrix but not of type Symmetric, would
	// be better to use EigenSym here but hard to convert.
	var ei mat64.Eigen
	ok := ei.Factorize(q3, false, true)
	if !ok {
		panic("Can't factorize covariance of means matrix\n")
	}
	dir := ei.Vectors()

	dir.Solve(q1.T(), dir)

	// Convert the eigenvalues to real
	evc := ei.Values(nil)
	ev := make([]float64, ndir)
	for i, x := range evc[0:ndir] {
		if math.Abs(imag(x)) > 1e-5 {
			panic("imaginary eigenvalues")
		}
		ev[i] = real(x)
	}
	sir.Eig = ev

	// Unpack the direction vectors
	for j := 0; j < ndir; j++ {
		sir.Dir = append(sir.Dir, mat64.Col(nil, j, dir))
	}

	// If needed, convert to original basis
	if sir.projBasis != nil {
		p := sir.Data.NumCov()
		for j := 0; j < ndir; j++ {
			b := make([]float64, p)
			v := mat64.NewVector(p, b)
			u := mat64.NewVector(len(sir.Dir[j]), sir.Dir[j])
			v.MulVec(sir.projBasis, u)
			sir.Dir[j] = b
		}
	}
}

func resizeInt(x []int, n int) []int {
	if cap(x) < n {
		x = make([]int, n)
	}
	return x[0:n]
}
