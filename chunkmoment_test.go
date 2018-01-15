package dimred

import (
	"bufio"
	"fmt"
	"math"
	"math/rand"
	"os"
	"testing"

	"gonum.org/v1/gonum/floats"

	"github.com/kshedden/dstream/dstream"
)

func cmdat1(chunksize int) (dstream.Dstream, dstream.Reg) {

	n := 100
	p := 3
	r := 0.6
	rc := math.Sqrt(1 - r*r)
	da := make([][]float64, p+2)
	rand.Seed(34234)

	// Covariates
	for j := 0; j < p+2; j++ {
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

	// Use to define segments
	for i := 0; i < n; i++ {
		da[p+1][i] = math.Floor(math.Sqrt(float64(i)))
	}

	// Outcome
	for i := 0; i < n; i++ {
		u := da[1][i] + 2*da[2][i] - 2*da[3][i]
		if u > 1 {
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
	na = append(na, "seg")
	dp := dstream.NewFromArrays(ida, na)
	dp = dstream.Segment(dp, []string{"seg"})
	dp = dstream.DropCols(dp, []string{"seg"})
	na = dp.Names()
	rdp := dstream.NewReg(dp, "y", na[1:len(na)], "", "")

	return dp, rdp
}

func TestCM1(t *testing.T) {

	_, rdp := cmdat1(20)

	if true {
		fid, err := os.Create("tmp.csv")
		if err != nil {
			panic(err)
		}
		wtr := bufio.NewWriter(fid)
		dstream.ToCSV(rdp).SetWriter(wtr).Done()
		wtr.Flush()
		fid.Close()
	}

	cm := &chunkMoment{Data: rdp}
	cm.walk()

	if cm.ny[0] != 69 || cm.ny[1] != 31 {
		t.Fail()
	}
	if cm.ntot != 100 {
		t.Fail()
	}
	if cm.nchunk != 10 {
		t.Fail()
	}

	// Check chunk sizes
	for k, v := range cm.chunkn {
		n := 0
		for _, u := range v {
			n += u
		}
		if n != 2*k+1 {
			t.Fail()
		}
	}

	// Check marginal mean
	cm.calcMargMean()
	mn := cm.MargMean()
	emn := []float64{-0.109787, 0.031292, -0.034271}
	if !floats.EqualApprox(mn, emn, 0.001) {
		t.Fail()
	}

	// Check conditional means
	ymns := [][]float64{
		[]float64{-0.456924, -0.265470, -0.034958},
		[]float64{0.662873, 0.691826, -0.032740},
	}
	for y := 0; y < 2; y++ {
		ymn := cm.YMean(y)
		if !floats.EqualApprox(ymn, ymns[y], 0.001) {
			t.Fail()
		}
	}

	// Check marginal covariance
	cm.calcMargCov()
	mc := cm.MargCov()
	emc := []float64{1.21597913, 0.69747121, 0.64119799, 0.69747121, 0.93473454, 0.70314174,
		0.64119799, 0.70314174, 1.0095865}
	if !floats.EqualApprox(mc, emc, 0.001) {
		t.Fail()
	}

	// Check conditional covariances
	ycovs := [][]float64{
		[]float64{0.83819425, 0.45300817, 0.5069568, 0.45300817, 0.7105614, 0.64775668,
			0.5069568, 0.64775668, 0.91489262},
		[]float64{1.19163166, 0.50193346, 0.93827903, 0.50193346, 0.80137311, 0.824953,
			0.93827903, 0.824953, 1.22035334},
	}
	for y := 0; y < 2; y++ {
		ycov := cm.YCov(y)
		if !floats.EqualApprox(ycov, ycovs[y], 0.001) {
			t.Fail()
		}
	}
}
