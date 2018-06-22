package dimred

import (
	"math"
	"math/rand"
	"testing"
	"time"

	"github.com/kshedden/dstream/dstream"
)

// genred generates a mxn matrix in which columns k1, k1+1 are proportional and
// k2, k2+1 are proportional.  The rank of the resulting matrix is n-2..
func genred(m, n, k1, k2 int) []float64 {

	rand.Seed(time.Now().UTC().UnixNano())

	a := make([]float64, m*n)

	for j := 0; j < m*n; j++ {
		a[j] = rand.NormFloat64()
	}

	for i := 0; i < m; i++ {
		a[i*n+k1+1] = 2 * a[i*n+k1]
	}

	for i := 0; i < m; i++ {
		a[i*n+k2+1] = 2 * a[i*n+k2]
	}
	return a
}

// cprod returns the nxn Gram matrix of the given mxn matrix.
func cprod(a []float64, m, n int) []float64 {

	b := make([]float64, n*n)

	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			u := 0.0
			for k := 0; k < m; k++ {
				u += a[k*n+i] * a[k*n+j]
			}
			b[i*n+j] = u
		}
	}

	return b
}

func TestFrank1(t *testing.T) {

	n := 10
	p := 5

	a := genred(n, p, 1, 3)
	b := cprod(a, n, p)

	pos, ell := frank(b, p, 1e-6)

	m := len(pos)
	for j := 0; j < p; j++ {
		for k := 0; k <= j; k++ {
			cp := 0.0
			for i := 0; i < m; i++ {
				cp += ell[i*p+j] * ell[i*p+k]
			}
			if math.Abs(b[j*p+k]-cp) > 1e-6 {
				t.Fail()
			}
		}
	}

	if len(pos) != 3 {
		t.Fail()
	}

	vpos := make(map[int]bool)
	for _, k := range pos {
		vpos[k] = true
	}

	if vpos[1] && vpos[2] {
		t.Fail()
	}
}

func TestFullRank1(t *testing.T) {

	x1 := []interface{}{
		[]float64{0, 0, 0},
		[]float64{1, 1, 1},
		[]float64{2, 2, 3},
	}
	x2 := []interface{}{
		[]float64{1, 1, 1},
		[]float64{1, 1, 1},
		[]float64{1, 1, 1},
	}
	x3 := []interface{}{
		[]float64{0, 0, 0},
		[]float64{2, 2, 2},
		[]float64{4, 4, 6},
	}
	x4 := []interface{}{
		[]float64{0, 0, 0},
		[]float64{1, 1, 1},
		[]float64{2, 2, 3},
	}
	dat := [][]interface{}{x1, x2, x3, x4}
	na := []string{"x1", "x2", "x3", "x4"}
	da := dstream.NewFromArrays(dat, na)

	fr := NewFullRank(da).Keep("x4").Done()
	rd := fr.Data()

	enames := []string{"x2", "x3", "x4"}
	for i := range enames {
		if enames[i] != rd.Names()[i] {
			t.Fail()
		}
	}

	fr = NewFullRank(da).Done()
	rd = fr.Data()

	enames = []string{"x2", "x3"}
	for i := range enames {
		if enames[i] != rd.Names()[i] {
			t.Fail()
		}
	}
}
