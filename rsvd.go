package dimred

// Background information for the algorithm implemented here:
// https://web.stanford.edu/group/mmds/slides2010/Martinsson.pdf

import (
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// A sparse matrix.
type SPM struct {
	row  []int
	col  []int
	dat  []float64
	nrow int
	ncol int
}

// NewSPM creates a sparse matrix of shape nrow x ncol.  For each i,
// dat[i] is inserted into the matrix at position row[i], col[i].  The
// data values can occur in arbitrary order, but (row, column) index
// pairs should not be repeated.
func NewSPM(row, col []int, dat []float64, nrow, ncol int) *SPM {
	return &SPM{
		row:  row,
		col:  col,
		dat:  dat,
		nrow: nrow,
		ncol: ncol,
	}
}

// T returns the transpose of a sparse matrix.  It is a shallow copy
// that shares the same data and index arrays as its source.
func (r *SPM) T() *SPM {

	return &SPM{
		row:  r.col,
		col:  r.row,
		dat:  r.dat,
		nrow: r.ncol,
		ncol: r.nrow,
	}
}

// RSVD approximately computes the SVD using a randomized SVD
// algorithm.
type RSVD struct {
	inmat *SPM
	nfac  int

	// The calculated SVD
	u      *mat.Dense
	v      *mat.Dense
	values []float64
}

// UTo places the U factor of the SVD (USV') into the destination
// matrix.  If the destination matrix is nil, a new matrix is
// allocated and the U factor of the SVD is copied into it.  In either
// case, the U factor is returned.
func (r *RSVD) UTo(dst *mat.Dense) *mat.Dense {

	if dst == nil {
		return r.u
	}

	r.u.Copy(dst)
	return dst
}

// VTo places the V factor of the SVD (USV') into the destination
// matrix.  If the destination matrix is nil, a new matrix is
// allocated and the U factor of the SVD is copied into it.  In either
// case, the U factor is returned.
func (r *RSVD) VTo(dst *mat.Dense) *mat.Dense {

	if dst == nil {
		return r.v
	}

	r.v.Copy(dst)
	return dst
}

// Values places the diagonal elements of the matrix S in the SVD
// (USV') into the distination slice.  If the destination slice is
// nil, a new slice is allocated and the diagonal elements of S are
// copied into it.  In either case, the slice containing the elements
// of S is returned.
func (r *RSVD) Values(s []float64) []float64 {

	if s == nil {
		return r.values
	}

	copy(s, r.values)
	return s
}

// leftmul places the matrix product of a sparse matrix with a dense
// matrix into a dense matrix.  The result follows c = a*b.
func leftmul(a *SPM, b *mat.Dense, c *mat.Dense) {

	// zero the result
	crow, ccol := c.Dims()
	for i := 0; i < crow; i++ {
		for j := 0; j < ccol; j++ {
			c.Set(i, j, 0)
		}
	}

	for f := range a.row {
		i, j := a.row[f], a.col[f]
		z := a.dat[f]
		for k := 0; k < ccol; k++ {
			c.Set(i, k, c.At(i, k)+z*b.At(j, k))
		}
	}
}

// Factorize uses the RSVD algorithm to obtain an approximation to the
// provided matrix inmat.  The approximate SVD is truncated to nfac
// factors.
func (r *RSVD) Factorize(inmat *SPM, nfac, npow int) {

	r.inmat = inmat
	r.nfac = nfac
	nrow := inmat.nrow
	ncol := inmat.ncol

	// The Gaussian projection matrix.
	om := mat.NewDense(ncol, nfac, nil)
	for i := 0; i < ncol; i++ {
		for j := 0; j < nfac; j++ {
			om.Set(i, j, rand.NormFloat64())
		}
	}

	// Randomly project the data matrix.
	ymat := mat.NewDense(nrow, nfac, nil)
	leftmul(inmat, om, ymat)
	ym := mat.NewDense(ncol, nfac, nil)
	for k := 0; k < npow; k++ {
		leftmul(inmat.T(), ymat, ym)
		leftmul(inmat, ym, ymat)
	}

	// QR decomposition of the projected matrix.  Gonum doesn't
	// have thin QR so use thin SVD instead.
	var qsv mat.SVD
	ok := qsv.Factorize(ymat, mat.SVDThin)
	if !ok {
		panic("svd!")
	}
	qdat := make([]float64, nrow*nfac)
	qmat := mat.NewDense(nrow, nfac, qdat)
	qsv.UTo(qmat)

	// b' = a'q
	btmat := mat.NewDense(ncol, nfac, nil)
	leftmul(r.inmat.T(), qmat, btmat)

	// b' = v s u'
	sv := new(mat.SVD)
	ok = sv.Factorize(btmat, mat.SVDThin)
	if !ok {
		panic("svd!\n")
	}
	uhmat := sv.VTo(nil)

	r.v = sv.UTo(nil)

	r.u = mat.NewDense(nrow, nfac, nil)
	r.u.Mul(qmat, uhmat)

	r.values = sv.Values(nil)
	for j := range r.values {
		r.values[j] = math.Pow(r.values[j], 1/float64(2*npow+1))
	}

}
