package dimred

import (
	"math/rand"
	"testing"
)

func TestRSVD1(t *testing.T) {

	n := 1000
	p := 20

	var row, col []int
	var dat []float64
	var fdat []float64
	for i := 0; i < n; i++ {
		for j := 0; j < p; j++ {
			row = append(row, i)
			col = append(col, j)
			f := 1 + float64(j*j)
			z := f * rand.NormFloat64()
			dat = append(dat, z)
			fdat = append(fdat, z)
		}
	}

	spm := NewSPM(row, col, dat, n, p)

	sv := new(RSVD)
	sv.Factorize(spm, 2, 3)
	_ = sv.UTo(nil)
}
