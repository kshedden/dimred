package dimred

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/kshedden/statmodel/dataprovider"
)

func docdat1(chunksize int) (dataprovider.Data, dataprovider.Reg) {

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
	dp := dataprovider.NewFromArrays(ida, na)
	dp = dataprovider.SizeChunk(dp, chunksize)
	rdp := dataprovider.NewReg(dp, "y", na[1:6], "", "")

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

func docdat2(chunksize int) (dataprovider.Data, dataprovider.Reg) {

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
	dp := dataprovider.NewFromArrays(ida, na)
	dp = dataprovider.SizeChunk(dp, chunksize)
	rdp := dataprovider.NewReg(dp, "y", na[1:6], "", "")

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
func docdat3(chunksize int) (dataprovider.Data, dataprovider.Reg) {

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
	dp := dataprovider.NewFromArrays(ida, na)
	dp = dataprovider.SizeChunk(dp, chunksize)
	rdp := dataprovider.NewReg(dp, "y", na[1:6], "", "")

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
