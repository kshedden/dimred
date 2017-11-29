package dimred

import (
	"fmt"
	"log"
	"math"
	"os"

	"github.com/gonum/floats"
	"github.com/gonum/matrix/mat64"
	"github.com/kshedden/dstream/dstream"
)

type DOC struct {
	chunkMoment

	// The dimension reduction direction obtained from the means.
	// Also, the standardized difference between the means for Y=1
	// and Y=0.
	meandir []float64

	// The dimension redution directions obtained from the covariances
	covdirs [][]float64

	// The standardized dimension reduction directions for the covariance difference
	stcovdirs [][]float64

	// Eigenvalues
	eig []float64

	// The standardized difference between the covariance matrices
	// for Y=1 and Y=0.
	stcovdiff []float64

	projDim int
}

// CovDir returns an estimated dimension reduction direction derived
// from the covariances.
func (doc *DOC) CovDir(j int) []float64 {

	if doc.projDim == 0 {
		return doc.covdirs[j]
	}

	return doc.invproject(doc.covdirs[j])
}

// MeanDir returns the estimated dimension reduction direction derived
// from the means.
func (doc *DOC) MeanDir() []float64 {

	if doc.projDim == 0 {
		return doc.meandir
	}

	return doc.invproject(doc.meandir)
}

func NewDOC(data dstream.Reg) *DOC {

	d := &DOC{
		chunkMoment: chunkMoment{
			Data: data,
		},
	}

	return d
}

func (doc *DOC) Done() *DOC {
	doc.walk()
	doc.calcMargMean()
	doc.calcMargCov()
	return doc
}

func (doc *DOC) SetProjection(ndim int) *DOC {
	doc.projDim = ndim
	return doc
}

func (doc *DOC) Fit(ndir int) {

	p := doc.Dim()
	if doc.projDim != 0 {
		doc.doProjection(doc.projDim)
		p = doc.projDim
	}
	pp := p * p

	margcov := mat64.NewSymDense(p, doc.GetMargCov())
	msr := new(mat64.Cholesky)
	if ok := msr.Factorize(margcov); !ok {
		print("Can't factorize marginal covariance")
		panic("")
	}

	// Calculate the standardized difference between the group
	// means, which is also the dimension reduction direction (in
	// the original coordinates) based on the mean.
	// TODO: what is the right standardization to use here?
	doc.meandir = make([]float64, p)
	floats.SubTo(doc.meandir, doc.GetMean(0), doc.GetMean(1))
	v := mat64.NewVector(p, doc.meandir)
	err := v.SolveVec(margcov, v)
	if err != nil {
		panic(err)
	}
	if doc.log != nil {
		doc.log.Printf("Mean-based dimension reduction direction/standardized mean difference:")
		doc.log.Printf("%v", doc.meandir)
	}

	// Caculate the standardized difference between the group
	// covariances.
	cd := make([]float64, pp)
	floats.SubTo(cd, doc.GetCov(0), doc.GetCov(1))
	cdm := mat64.NewSymDense(p, cd)
	q1 := new(mat64.TriDense)
	q1.LFromCholesky(msr)
	q2 := new(mat64.Dense)
	q2.Solve(q1, cdm)
	doc.stcovdiff = make([]float64, pp)
	q3 := mat64.NewDense(p, p, doc.stcovdiff)
	err = q3.Solve(q1, q2.T())
	if err != nil {
		panic("can't back-transform")
	}
	if doc.log != nil {
		doc.log.Printf("Standardized covariance difference:")
		doc.log.Printf(fmt.Sprintf("%v", doc.stcovdiff))
	}

	// Calculate the eigenvectors of the standardized covariance
	// difference
	es := new(mat64.EigenSym)
	m1 := mat64.NewSymDense(p, doc.stcovdiff)
	ok := es.Factorize(m1, true)
	if !ok {
		panic("can't factorize")
	}

	// Sort the absolute eigenvalues
	eig := es.Values(nil)
	eq := make([]float64, len(eig))
	for j, e := range eig {
		eq[j] = math.Abs(e)
	}
	ii := make([]int, len(eig))
	floats.Argsort(eq, ii)

	doc.stcovdirs = make([][]float64, ndir)
	m4 := new(mat64.Dense)
	m4.EigenvectorsSym(es)
	for j := 0; j < ndir; j++ {
		k := ii[len(ii)-j-1]
		doc.stcovdirs[j] = mat64.Col(nil, k, m4)
		doc.eig = append(doc.eig, eig[k])
	}
	if doc.log != nil {
		doc.log.Printf("Eigenvalues:\n")
		doc.log.Printf("%v", doc.eig)
	}

	// The DR direction obtained from the covariances
	doc.covdirs = make([][]float64, ndir)
	qm := new(mat64.TriDense)
	qm.LFromCholesky(msr)
	for j, v := range doc.stcovdirs {
		doc.covdirs[j] = make([]float64, p)
		u := mat64.NewVector(p, doc.covdirs[j])
		w := mat64.NewVector(p, v)
		err = u.SolveVec(qm.T(), w)
		if err != nil {
			print("can't back-transform covariance directions")
		}
	}
	if doc.log != nil {
		doc.log.Printf("Covariance directions")
		doc.log.Printf("%v", doc.covdirs)
	}
}

func (doc *DOC) SetLogFile(filename string) *DOC {
	fid, err := os.Create(filename)
	if err != nil {
		panic(err)
	}
	doc.log = log.New(fid, "", log.Lshortfile)

	return doc
}
