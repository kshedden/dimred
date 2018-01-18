package dimred

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"

	"github.com/kshedden/dstream/dstream"
)

func armat(d int, r float64) mat.Symmetric {

	c := make([]float64, d*d)

	for i := 0; i < d; i++ {
		for j := 0; j < d; j++ {
			c[i*d+j] = math.Pow(r, math.Abs(float64(i-j)))
		}
	}

	return mat.NewSymDense(d, c)
}

func gendat1(chunksize int) dstream.Dstream {

	n := 10000
	p := 5
	r := 0.6
	rc := math.Sqrt(1 - r*r)
	da := make([][]float64, p+1)

	// Noise
	for j := 0; j < p+1; j++ {
		if j == 0 {
			for i := 0; i < n; i++ {
				da[j] = append(da[j], rand.NormFloat64())
			}
		} else {
			for i := 0; i < n; i++ {
				da[j] = append(da[j], r*da[j-1][i]+rc*rand.NormFloat64())
			}
		}
	}

	// Signal
	for i := 0; i < n; i++ {
		da[0][i] = float64(i % 3)
		for j := 1; j < p+1; j++ {
			if j != 2 {
				da[j][i] += float64(i%3 + j)
			} else {
				da[j][i] += float64(j)
			}
		}
	}

	var ida [][]interface{}
	for _, x := range da {
		ida = append(ida, []interface{}{x})
	}
	na := []string{"y"}
	for j := 0; j < p; j++ {
		na = append(na, fmt.Sprintf("x%d", j+1))
	}
	dp := dstream.NewFromArrays(ida, na)
	dp = dstream.MaxChunkSize(dp, chunksize)

	return dp
}

func TestSIR1(t *testing.T) {

	dp := gendat1(1000)

	f := func(y float64) int {
		return int(y)
	}

	sir := NewSIR(dp, "y", f).Done()

	// Check the slice means
	for j, v := range [][]float64{
		[]float64{1, 2, 3, 4, 5},
		[]float64{2, 2, 4, 5, 6},
		[]float64{3, 2, 5, 6, 7},
	} {
		if !floats.EqualApprox(sir.SliceMeans()[j], v, 0.05) {
			fmt.Printf("Slice means do not match:\n")
			fmt.Printf("%v\n", sir.SliceMeans()[j])
			fmt.Printf("%v\n", v)
			t.Fail()
		}
	}

	// Check that the marginal covariance is approximately AR(0.6)
	mcc := new(mat.Dense)
	mcc.Sub(sir.MargCov(), sir.CovMean())
	dm := new(mat.Dense)
	dm.Sub(mcc, armat(5, 0.6))
	dmx := mat.Norm(dm, 2)
	if dmx > 0.1 {
		fmt.Printf("Marginal covariances do not match: %f\n", dmx)
		t.Fail()
	}

	// Check slice sizes
	ss := []int{3334, 3333, 3333}
	for i, x := range sir.SliceSizes() {
		if x != ss[i] {
			msg := fmt.Sprintf("Slice sizes don't match, %d != %d\n", x, ss[i])
			print(msg)
			t.Fail()
		}
	}
}

func gendat2(chunksize int) dstream.Dstream {

	n := 10000
	p := 5
	r := 0.6
	rc := math.Sqrt(1 - r*r)
	da := make([][]float64, p+1)

	// Noise
	for i := 0; i < n; i++ {
		da[0] = append(da[0], rand.NormFloat64())
	}
	for j := 1; j < p+1; j++ {
		for i := 0; i < n; i++ {
			da[j] = append(da[j], r*da[j-1][i]+rc*rand.NormFloat64())
		}
	}

	// Signal
	for i := 0; i < n; i++ {
		da[0][i] = 0
		for j := 1; j < p+1; j++ {
			da[0][i] += float64(j%3) * da[j][i]
		}
	}

	// Names
	na := []string{"y"}
	for j := 0; j < p; j++ {
		na = append(na, fmt.Sprintf("x%d", j+1))
	}

	var ida [][]interface{}
	for _, x := range da {
		ida = append(ida, []interface{}{x})
	}
	dp := dstream.NewFromArrays(ida, na)
	dp = dstream.MaxChunkSize(dp, chunksize)

	return dp
}

func TestSIR2(t *testing.T) {

	dp := gendat2(1000)

	f := func(y float64) int {
		ii := int(5*y + 100)
		return ii
	}

	sir := NewSIR(dp, "y", f).NDir(2).Done()

	// Check that the marginal covariance is approximately AR(0.6)
	dm := new(mat.Dense)
	dm.Sub(sir.MargCov(), armat(5, 0.6))
	dmx := mat.Norm(dm, 2)
	if dmx > 0.1 {
		fmt.Printf("Marginal covariances do not match: %f\n", dmx)
		t.Fail()
	}

	sir.Fit()

	rq := []float64{2, 0, 1, 2}
	for j := 1; j < 5; j++ {
		r := sir.Dir[0][j] / sir.Dir[0][0]
		if math.Abs(rq[j-1]-r) > 0.01 {
			fmt.Printf("EDR direction does not agree\n")
			t.Fail()
		}
	}
}

func gendat3(chunksize int) dstream.Dstream {

	n := 10000
	p := 5
	r := 0.6
	rc := math.Sqrt(1 - r*r)
	da := make([][]float64, p+1)

	// Noise
	for i := 0; i < n; i++ {
		da[0] = append(da[0], rand.NormFloat64())
	}
	for j := 1; j < p+1; j++ {
		for i := 0; i < n; i++ {
			da[j] = append(da[j], r*da[j-1][i]+rc*rand.NormFloat64())
		}
		if j == p {
			// Final direction will be dropped in projection
			for i := 0; i < n; i++ {
				da[j][i] /= 10
			}
		}
	}

	// Signal
	for i := 0; i < n; i++ {
		da[0][i] = 0
		for j := 1; j < p+1; j++ {
			da[0][i] += float64(j%3) * da[j][i]
		}
	}

	// Variable names
	na := []string{"y"}
	for j := 0; j < p; j++ {
		na = append(na, fmt.Sprintf("x%d", j+1))
	}

	var ida [][]interface{}
	for _, x := range da {
		ida = append(ida, []interface{}{x})
	}
	dp := dstream.NewFromArrays(ida, na)
	dp = dstream.MaxChunkSize(dp, chunksize)

	return dp
}

func TestSIR3(t *testing.T) {

	dp := gendat3(1000)

	f := func(y float64) int {
		ii := int(5*y + 100)
		return ii
	}

	sir := NewSIR(dp, "y", f).SetLogFile("tt").Done()
	sir.ProjectEigen(4)
	sir.Fit()
}
