package dimred

import (
	"math"
	"sort"

	"gonum.org/v1/gonum/floats"

	"github.com/kshedden/dstream/dstream"
)

// NewFullRank returns a FullRank struct for the given dstream.  All columns
// not specified in a call to Keep must have float64 type.
func NewFullRank(data dstream.Dstream) *FullRank {

	return &FullRank{
		data: data,
	}
}

// FullRank eliminates linear dependencies among columns in a dstream.  A rank-revealing
// Cholesky decomposition of the Gram matrix is used to do this.
type FullRank struct {

	// The Gram matrix of the columns being assessed for linear dependence
	cpr []float64

	// The input dstream
	data dstream.Dstream

	// The output dstream
	rdata dstream.Dstream

	// Variables that we will keep regardless (ignore in the check for linear dependence).
	keep    []string
	keeppos []int

	// Positions to consider in the check for linear dependence (everything not in keeppos).
	checkpos []int
}

func (fr *FullRank) init() {

	vpos := make(map[string]int)
	for k, v := range fr.data.Names() {
		vpos[v] = k
	}

	km := make(map[string]bool)
	for _, s := range fr.keep {
		fr.keeppos = append(fr.keeppos, vpos[s])
		km[s] = true
	}

	for k, n := range fr.data.Names() {
		if !km[n] {
			fr.checkpos = append(fr.checkpos, k)
		}
	}
}

// getcpr calculates the cross product matrix of the variables being checked.
func (fr *FullRank) getcpr() {

	p := len(fr.keeppos)
	cpr := make([]float64, p*p)

	fr.data.Reset()
	for fr.data.Next() {
		vars := make([][]float64, p)

		for j, k := range fr.keeppos {
			vars[j] = fr.data.GetPos(k).([]float64)
		}

		for j1 := 0; j1 < p; j1++ {
			for j2 := 0; j2 <= j1; j2++ {
				u := floats.Dot(vars[j1], vars[j2])
				cpr[j1*p+j2] += u
				if j1 != j2 {
					cpr[j2*p+j1] += u
				}
			}
		}
	}

	fr.cpr = cpr
}

// Done completes configuration.
func (fr *FullRank) Done() *FullRank {

	fr.init()
	fr.getcpr()

	p := len(fr.keeppos)
	pos, _ := frank(fr.cpr, p, 1e-6)

	rpos := make(map[int]bool)
	for _, k := range pos {
		rpos[fr.checkpos[k]] = true
	}

	for _, k := range fr.keeppos {
		rpos[k] = true
	}

	var drop []string
	names := fr.data.Names()
	for k := range rpos {
		if !rpos[k] {
			drop = append(drop, names[k])
		}
	}

	fr.rdata = dstream.DropCols(fr.data, names...)

	return fr
}

// Data returns the dstream constructed by FullRank.
func (fr *FullRank) Data() dstream.Dstream {
	return fr.rdata
}

// Keep specifies variables that are not considered in the linear
// independence assessment.
func (fr *FullRank) Keep(vars []string) *FullRank {

	fr.keep = vars
	return fr
}

// fr returns the positions of a maximal subset of rows/columns of the symmetric
// nxn matrix a that are linearly independent.
func frank(a []float64, n int, tol float64) ([]int, []float64) {

	// Page 7:
	// Note that the update for d is wrong
	// http://www.mathe.tu-freiberg.de/naspde2010/sites/default/files/harbrecht.pdf

	// Permutation
	perm := make([]int, n)
	for i := range perm {
		perm[i] = i
	}

	d := make([]float64, n)
	for i := 0; i < n; i++ {
		d[i] = a[i*n+i]
	}

	err := 0.0
	for i := 0; i < n; i++ {
		if d[i] < -1e-6 {
			panic("negative..\n")
		}
		err += d[i]
	}

	ell := make([]float64, n*n)

	m := 0
	for err > tol {

		dm := d[perm[m]]
		i := m
		for j := m + 1; j < n; j++ {
			b := d[perm[j]]
			if b > dm {
				dm = b
				i = j
			}
		}

		perm[m], perm[i] = perm[i], perm[m]

		ell[m*n+perm[m]] = math.Sqrt(d[perm[m]])

		for i := m + 1; i < n; i++ {
			u := 0.0
			for j := 0; j < m; j++ {
				u += ell[j*n+perm[m]] * ell[j*n+perm[i]]
			}
			ell[m*n+perm[i]] = (a[perm[m]*n+perm[i]] - u) / ell[m*n+perm[m]]

			d[perm[i]] -= ell[m*n+perm[i]] * ell[m*n+perm[i]]
		}

		err = 0
		for i := m + 1; i < n; i++ {
			if d[perm[i]] < -1e-6 {
				panic("negative")
			}
			err += d[perm[i]]
		}

		m++
	}

	perm = perm[0:m]
	sort.Sort(sort.IntSlice(perm))

	return perm[0:m], ell[0 : m*n]
}

/*
	fmt.Printf("perm=%v\n", perm)
	fmt.Printf("m=%d\n", m)

	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			lcp := 0.0
			for k := 0; k < m; k++ {
				lcp += ell[k*n+i] * ell[k*n+j] //??
			}
			fmt.Printf("%f %f\n", lcp, a[i*n+j])
		}
	}
}
*/
