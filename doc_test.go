package dimred

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/brookluers/dstream/dstream"
)

// arspec specifies a test population for simulation.  The
// observations alternate between the two groups.
type arspec struct {

	// Sample size
	n int

	// Number of variables
	p int

	// Autocorrelation parameter
	r float64

	// Transformation function that can be used to introduce a
	// mean. The arguments are i (subject), j (variable) and e,
	// where e is the AR-1 error term.
	xform func(int, int, float64) float64
}

func docdat1(chunksize int, ars arspec) (dstream.Dstream, dstream.Reg) {

	n := ars.n
	p := ars.p
	r := ars.r
	rc := math.Sqrt(1 - r*r)
	da := make([][]float64, p+1) // first element is group label 0/1

	// Autocorrelated noise
	for j := 0; j < p+1; j++ {
		if j == 0 {
			// First variable is all white
			for i := 0; i < n; i++ {
				da[j] = append(da[j], rand.NormFloat64())
			}
		} else {
			// Subsequent variables are correlated with previous variables
			for i := 0; i < n; i++ {
				da[j] = append(da[j], r*da[j-1][i]+rc*rand.NormFloat64())
			}
		}
	}
	// Mean structure
	if ars.xform != nil {
		for i := 0; i < n; i++ {
			for j := 1; j < p+1; j++ {
				da[j][i] = ars.xform(i, j, da[j][i])
			}
		}
	}

	// Create the group labels
	for i := 0; i < n; i++ {
		da[0][i] = float64(i % 2)
	}

	// Create a dstream
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
	rdp := dstream.NewReg(dp, "y", na[1:6], "", "")

	return dp, rdp
}

func TestDOC1(t *testing.T) {

	xform := func(i int, j int, x float64) float64 {
		if j == 0 {
			return float64(i % 2)
		}
		z := float64(i%2)*float64(j) + float64(i%2+1)*x
		if i%2 == 0 {
			z *= 2
		}
		return z
	}

	ars := arspec{n: 10000, p: 5, r: 0.6, xform: xform}

	_, rdp := docdat1(1000, ars)

	doc := NewDOC(rdp).SetLogFile("ss").Done()
	doc.Fit(4)

	_ = doc // just a smoke test
}

func TestDOC2(t *testing.T) {

	ars := arspec{n: 10000, p: 5, r: 0.6}
	_, rdp := docdat1(1000, ars)

	doc := NewDOC(rdp).SetLogFile("s2").Done()
	doc.Fit(4)

	_ = doc // Just a smoke test
}

// Generate data from a forward regression model.  X marginally is AR,
// Y|X depends only on one linear function of X.
func docdat3(chunksize int) (dstream.Dstream, dstream.Reg) {

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

	for i := 0; i < n; i++ {
		u := da[1][i] - 2*da[2][i] + 3*da[4][i]
		if u > 0 {
			da[0][i] = 1
		} else {
			da[0][i] = 0
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
	rdp := dstream.NewReg(dp, "y", na[1:6], "", "")

	return dp, rdp
}

func TestDOC3(t *testing.T) {

	_, rdp := docdat3(1000)

	doc := NewDOC(rdp).SetLogFile("s3").Done()
	doc.Fit(2)

	_ = doc // Just a smoke test
}

func TestProj(t *testing.T) {

	xform := func(i, j int, x float64) float64 {
		if j <= 2 {
			return 2 * x
		}
		return x
	}

	ars := arspec{n: 10000, p: 5, r: 0.6, xform: xform}
	_, rdp := docdat1(1000, ars)

	doc := NewDOC(rdp).SetLogFile("sp").SetProjection(2).Done()
	doc.Fit(2)
}

func TestMeandir(t *testing.T){

     mult := 3.0 
     y := []interface{}{
       []float64{1.0, 1.0},
       []float64{1.0, 0.0, 0.0},
       []float64{0.0},
     }
     x1 := []interface{}{
     	[]float64{-1.0 * mult, -1.0 * mult},
	[]float64{1.0 * mult, 1.0 * mult, -1.0*mult},
	[]float64{1.0 * mult},
     }
     x2 := []interface{}{
     	[]float64{-1.0 * mult, 1.0 * mult},
	[]float64{-1.0 * mult, 1.0 * mult, 1.0 * mult},
	[]float64{-1.0 * mult},
     }
     dat := [][]interface{}{x1, x2, y}
     na := []string{"x1", "x2", "y"}
     ds := dstream.NewFromArrays(dat, na)
     dr := dstream.NewReg(ds, "y", nil, "", "")
     doc := NewDOC(dr)
     doc.Init()
     doc.Fit(2)
     // Marginal covariance:
     // (9   -3
     //  -3   9 )
     truecov := []float64{9.0, -3.0, -3.0, 9.0}
     // Inverse of marginal covariance: 
     // (9/72   3/72
     //  3/72   9/72)
     mean0 := doc.GetMean(0)
     mean1 := doc.GetMean(1)
     margcov := doc.MargCov()
     fmt.Printf("GetMean(0) = %v\nGetMean(1) = %v\n", mean0, mean1)
     rawdiff := make([]float64, 2)
     rawdiff[0] = mean1[0] - mean0[0]
     rawdiff[1] = mean1[1] - mean0[1]
     fmt.Printf("Raw difference in means = %v\n", rawdiff)
     fmt.Printf("MargCov = %v\n", margcov)
     fmt.Printf("MeanDir = %v\n", doc.MeanDir())
     const TOL = 0.0000001
     answer := []float64{1.0 / 3.0, 1.0 / 3.0}
     md := doc.MeanDir()
     if (math.Abs(md[0] - answer[0]) > TOL || math.Abs(md[1] - answer[1]) > TOL){
     	t.Logf("mean direction incorrect\n")
	t.Fail()
     }
     for i := 0; i < len(margcov); i++{
     	 if (math.Abs(margcov[i] - truecov[i]) > TOL) {
	    t.Logf("marginal covariance incorrect\n")
	    t.Fail()
	 }
     }
}