package dimred

import (
	"fmt"
	"log"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"

	"github.com/kshedden/dstream/dstream"
)

// chunkMoment calculates first and second moments (means and
// covariances) for streaming data, stratifying by grouping variable.
type chunkMoment struct {

	// The data used to perform the analysis
	Data dstream.Reg

	// Sample size, mean and covariance, separately by y level, but
	// marginal over chunks.
	ny   []int
	mean [][]float64
	cov  [][]float64

	ntot   int // overall sample size
	nchunk int // number of chunks

	chunkmeans [][][]float64
	chunkn     [][]int

	// Marginal mean and covariance
	margmean []float64
	margcov  []float64

	// The means and covariances may be projected against this
	// basis before returning. If projDim = 0, projBasis should be
	// ignored.
	projBasis mat.Matrix
	projDim   int

	log *log.Logger
}

func (cm *chunkMoment) MargCov() []float64 {
	return cm.margcov
}

func (cm *chunkMoment) MargMean() []float64 {
	return cm.margmean
}

func (cm *chunkMoment) YCov(y int) []float64 {
	return cm.cov[y]

}

func (cm *chunkMoment) YMean(y int) []float64 {
	return cm.mean[y]
}

// setProjection uses the dominant eigenvectors of the marginal
// covariance matrix to project all calculated moments to a reduced
// space.
func (cm *chunkMoment) doProjection(ndim int) {

	// Restores the original full covariances.
	if ndim == 0 {
		cm.projDim = 0
		cm.projBasis = nil
	}

	p := cm.Data.NumCov()
	mcov := cm.margcov

	es := new(mat.EigenSym)
	ok := es.Factorize(mat.NewSymDense(p, mcov), true)
	if !ok {
		panic("unable to determine eigenvectors of marginal covariance")
	}

	evec := new(mat.Dense)
	evec.EigenvectorsSym(es)
	if cm.log != nil {
		cm.log.Printf("Eigenvalues of marginal covariance for projection:")
		cm.log.Printf(fmt.Sprintf("%v\n", es.Values(nil)))
		cm.log.Printf(fmt.Sprintf("Retaining %d-dimensional eigenspace, dropping %d dimensions\n", ndim, p-ndim))
	}
	evecv := evec.Slice(0, p, 0, ndim)

	cm.projBasis = evecv
	cm.projDim = ndim
}

// zero an array
func zero(x []float64) {
	for j := range x {
		x[j] = 0
	}
}

// zero and truncate a stack of workspaces
func zeroStack(x [][]float64) [][]float64 {
	x = x[0:cap(x)]
	for _, v := range x {
		zero(v)
	}
	return x[0:0]
}

// addBetweenCov adds the covariance of between-chunk means to the
// current value of cov. Everything is stratified by y level.
func (cm *chunkMoment) addBetweenCov() {

	p := cm.Data.NumCov()
	pp := p * p
	wk := make([]float64, pp)

	for yi := 0; yi < len(cm.mean); yi++ {

		if cm.ny[yi] == 0 {
			continue
		}

		zero(wk)

		for j, chunk := range cm.chunkmeans {
			for k1 := 0; k1 < p; k1++ {
				if len(chunk) <= yi {
					continue
				}
				u1 := chunk[yi][k1] - cm.mean[yi][k1]
				for k2 := 0; k2 <= k1; k2++ {
					u2 := chunk[yi][k2] - cm.mean[yi][k2]
					wk[k1*p+k2] += float64(cm.chunkn[j][yi]) * u1 * u2
				}
			}
		}

		reflect(wk, p)
		floats.Scale(1/float64(cm.ny[yi]), wk)
		floats.AddScaled(cm.cov[yi], 1, wk)
	}
}

// chunkMean calculates the means of the current chunk's data,
// stratifying by y-level.  The sample sizes per y-level are also
// calculated and returned.
func (cm *chunkMoment) chunkMean() ([][]float64, []int) {

	p := cm.Data.NumCov()
	y := cm.Data.YData()
	var mn [][]float64
	var ny []int

	// Calculate the sums and sample sizes
	for j := 0; j < p; j++ {
		x := cm.Data.XData(j)
		for i := 0; i < len(y); i++ {
			yi := int(y[i])
			mn = growf(mn, yi, p)
			mn[yi][j] += x[i]
			if j == 0 {
				ny = growi(ny, yi)
				ny[yi]++
			}
		}
	}

	// Normalize to get the means.
	for i, v := range mn {
		if ny[i] > 0 {
			floats.Scale(1/float64(ny[i]), v)
		}
	}

	return mn, ny
}

// growf ensures that dat is large enough so that dat[y] exists and
// holds a float64 slice of length p.
func growf(dat [][]float64, y, p int) [][]float64 {
	for len(dat) <= y {
		dat = append(dat, make([]float64, p))
	}
	return dat
}

// growi ensures that cov is large enough so that cov[y] exists.
func growi(ar []int, y int) []int {
	for len(ar) <= y {
		ar = append(ar, 0)
	}
	return ar
}

// chunkCov calculates the covariances of the current chunk's data,
// stratifying by y-level.  The provided storage is used, and grown if
// needed.
func (cm *chunkMoment) chunkCov(cov [][]float64, mn [][]float64, ny []int) [][]float64 {

	p := cm.Data.NumCov()
	pp := p * p
	cov = zeroStack(cov)
	y := cm.Data.YData()

	// Fill in one triangle of the covariance.
	for j1 := 0; j1 < p; j1++ {
		x1 := cm.Data.XData(j1)
		for j2 := 0; j2 <= j1; j2++ {
			x2 := cm.Data.XData(j2)
			for i := 0; i < len(y); i++ {
				yi := int(y[i])
				cov = growf(cov, yi, pp)
				cov[yi][j1*p+j2] += (x1[i] - mn[yi][j1]) * (x2[i] - mn[yi][j2])
			}
		}
	}

	// Fill in the other triangle and scale by sample size.
	for i, v := range cov {
		reflect(v, p)
		if ny[i] > 0 {
			floats.Scale(1/float64(ny[i]), v)
		}
	}

	return cov
}

// Dim returns the dimension of the problem (the number of variables
// in all means and covariances).
func (cm *chunkMoment) Dim() int {
	if cm.projDim == 0 {
		return cm.Data.NumCov()
	}
	return cm.projDim
}

// conjugate takes an input matrix mat, and returns its conjugation B'
// * mat * B by the projection basis B.  mat must be p x p, where p is
// the number of variables.
func (cm *chunkMoment) conjugate(ma []float64) []float64 {
	q := cm.Data.NumCov() // Original dimension
	p := cm.projDim       // New dimension

	// The result will go here.
	mc := make([]float64, p*p)

	// Do the conjugation
	left := new(mat.Dense)
	left.Mul(cm.projBasis.T(), mat.NewDense(q, q, ma))
	r := mat.NewDense(p, p, mc)
	r.Mul(left, cm.projBasis)

	return mc
}

// project takes an input vector vec, and returns its projection B' *
// vec by the projection basis B.  The length of vec must be p, where
// p is the number of variables.
func (cm *chunkMoment) project(vec []float64) []float64 {

	q := cm.Data.NumCov() // Original dimension
	p := cm.projDim       // New dimension

	// The result will go here.
	mp := make([]float64, p)

	// Do the projection
	r := mat.NewVecDense(p, mp)
	r.MulVec(cm.projBasis.T(), mat.NewVecDense(q, vec))

	return mp
}

// invproject inverts the stored projection, mapping the vector vec
// back to the original coordinate system.
func (cm *chunkMoment) invproject(vec []float64) []float64 {

	q := cm.Data.NumCov() // Original dimension
	p := cm.projDim       // New dimension

	u := make([]float64, q)
	r := mat.NewVecDense(q, u)
	r.MulVec(cm.projBasis, mat.NewVecDense(p, vec))

	return u
}

// GetMean returns the conditional mean for y=i.  If a projection has
// been set, the projected conditional mean is returned.
func (cm *chunkMoment) GetMean(i int) []float64 {

	if cm.projDim == 0 {
		return cm.mean[i]
	}

	return cm.project(cm.mean[i])
}

// GetCov returns the conditional covariance for y=i.  If a projection
// has been set, the projected conditional covariance is returned.
func (cm *chunkMoment) GetCov(i int) []float64 {

	if cm.projDim == 0 {
		return cm.cov[i]
	}

	return cm.conjugate(cm.cov[i])
}

// Returns the marginal mean, possibly projected if a projection
// has been set.
func (cm *chunkMoment) GetMargMean() []float64 {

	if cm.projDim == 0 {
		return cm.margmean
	}

	return cm.project(cm.margmean)
}

// Returns the marginal covariance, possibly projected if a projection
// has been set.
func (cm *chunkMoment) GetMargCov() []float64 {

	if cm.projDim == 0 {
		return cm.margcov
	}

	return cm.conjugate(cm.margcov)
}

// walk cycles through the data and calculates chunk-wise summary
// statistics (mean and covariance by y level).
func (cm *chunkMoment) walk() {

	// Clear everything
	cm.Data.Reset()
	cm.ntot = 0
	cm.nchunk = 0
	cm.chunkn = cm.chunkn[0:0]
	cm.mean = cm.mean[0:0]
	cm.cov = cm.cov[0:0]
	cm.chunkmeans = cm.chunkmeans[0:0]
	cm.ny = cm.ny[0:0]

	p := cm.Data.NumCov()
	pp := p * p

	// A stack of p^2 dimensional workspaces.
	var cov [][]float64

	for cm.Data.Next() {

		y := cm.Data.YData()
		n := len(y)
		cm.ntot += n
		cm.nchunk++

		// Means for the current chunk (by y level)
		mn, ny := cm.chunkMean()

		// Covariances of current chunk (by y level)
		cov = cm.chunkCov(cov, mn, ny)

		// Update the counts
		for i, n := range ny {
			cm.ny = growi(cm.ny, i)
			cm.ny[i] += n
		}
		cm.chunkn = append(cm.chunkn, ny)

		// Update the means
		for i, v := range mn {
			cm.mean = growf(cm.mean, i, p)
			floats.AddScaled(cm.mean[i], float64(ny[i]), v)
		}

		// Update the mean covariances
		for i, v := range cov {
			cm.cov = growf(cm.cov, i, pp)
			floats.AddScaled(cm.cov[i], float64(ny[i]), v)
		}

		cm.chunkmeans = append(cm.chunkmeans, mn)
	}

	// Normalize the means
	for i, v := range cm.mean {
		floats.Scale(1/float64(cm.ny[i]), v)
	}

	// Normalize the mean covariances
	for i, v := range cm.cov {
		floats.Scale(1/float64(cm.ny[i]), v)
	}

	// Add Cov E[X|Y] to cm.cov, which currently holds E Cov[X|Y].
	cm.addBetweenCov()

	if cm.log != nil {
		cm.log.Printf("%d records", cm.ntot)
		cm.log.Printf("%d data chunks", cm.nchunk)
		cm.log.Printf("Sample sizes per chunk:")
		for j, chunkn := range cm.chunkn {
			cm.log.Printf(fmt.Sprintf("Chunk %6d %v", j, chunkn))
		}
		cm.log.Printf("Mean by y level:")
		for j, v := range cm.mean {
			cm.log.Printf("%d %v", j, v)
		}
		cm.log.Printf("Covariance by y level:")
		for j, v := range cm.cov {
			cm.log.Printf("%d %v", j, v)
		}
	}
}

// calcMargMean sets the margmean field, by collapsing over the y
// levels to get the marginal mean.
func (cm *chunkMoment) calcMargMean() {

	p := cm.Data.NumCov()
	margmean := make([]float64, p)

	for yi, v := range cm.mean {
		for k, u := range v {
			margmean[k] += float64(cm.ny[yi]) * u
		}
	}
	floats.Scale(1/float64(cm.ntot), margmean)
	if cm.log != nil {
		cm.log.Printf("Marginal mean:")
		cm.log.Printf(fmt.Sprintf("%v", margmean))
	}

	cm.margmean = margmean
}

// calcMargCov sets the margcov field, by collapsing over the y levels
// to get the marginal covariance matrix.
func (cm *chunkMoment) calcMargCov() {

	if cm.margmean == nil {
		cm.calcMargMean()
	}

	p := cm.Data.NumCov()
	pp := p * p

	// Mean within-y covariance
	wc := make([]float64, pp)
	for yi, v := range cm.cov {
		floats.AddScaled(wc, float64(cm.ny[yi]), v)
	}
	floats.Scale(1/float64(cm.ntot), wc)

	// Covariance of within-y means
	bc := make([]float64, pp)
	for yi, mn := range cm.mean {
		for k1 := 0; k1 < p; k1++ {
			u1 := mn[k1] - cm.margmean[k1]
			for k2 := 0; k2 <= k1; k2++ {
				u2 := mn[k2] - cm.margmean[k2]
				bc[k1*p+k2] += float64(cm.ny[yi]) * u1 * u2
			}
		}
	}
	reflect(bc, p)
	floats.Scale(1/float64(cm.ntot), bc)

	cm.margcov = make([]float64, pp)
	for i := 0; i < pp; i++ {
		cm.margcov[i] = bc[i] + wc[i]
	}

	if cm.log != nil {
		cm.log.Printf("Marginal covariance:")
		cm.log.Printf(fmt.Sprintf("%v", cm.margcov))
	}
}
