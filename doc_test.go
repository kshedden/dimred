package dimred

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/brookluers/dstream/dstream"
)

func docdat1(chunksize int) (dstream.Dstream, dstream.Reg) {

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
		da[0][i] = float64(i % 2)
		for j := 1; j < p+1; j++ {
			da[j][i] = float64(i%2)*float64(j) + float64(i%2+1)*da[j][i]
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
	dp = dstream.SizeChunk(dp, chunksize)
	rdp := dstream.NewReg(dp, "y", na[1:6], "", "")

	return dp, rdp
}

func TestDOC1(t *testing.T) {

	dp, rdp := docdat1(1000)
	_ = dp

	doc := &DOC{
		chunkMoment: chunkMoment{
			Data: rdp,
		},
	}

	doc.SetLogFile("ss")
	doc.Init()
	doc.Fit(4)

	_ = doc
}

func docdat2(chunksize int) (dstream.Dstream, dstream.Reg) {

	n := 10000
	p := 5
	r := 0.6
	rc := math.Sqrt(1 - r*r)
	da := make([][]float64, p+1)

	// AR noise
	for j := 0; j < p+1; j++ {
		if j == 0 || j == 3 {
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
		da[0][i] = float64(i % 2)

		// Covariance difference in two coordinates
		if i%2 == 0 {
			da[1][i] *= 2
			da[2][i] *= 2
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
	dp = dstream.SizeChunk(dp, chunksize)
	rdp := dstream.NewReg(dp, "y", na[1:6], "", "")

	return dp, rdp
}

func TestDOC2(t *testing.T) {

	dp, rdp := docdat2(1000)
	_ = dp

	doc := &DOC{
		chunkMoment: chunkMoment{
			Data: rdp,
		},
	}

	doc.SetLogFile("s2")
	doc.Init()
	doc.Fit(4)

	_ = doc
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
	dp = dstream.SizeChunk(dp, chunksize)
	rdp := dstream.NewReg(dp, "y", na[1:6], "", "")

	return dp, rdp
}

func TestDOC3(t *testing.T) {

	dp, rdp := docdat3(1000)
	_ = dp

	doc := &DOC{
		chunkMoment: chunkMoment{
			Data: rdp,
		},
	}

	doc.SetLogFile("s3")
	doc.Init()
	doc.Fit(2)

	_ = doc
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