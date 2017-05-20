package dimred

import (
	"fmt"
	"log"

	"github.com/gonum/floats"
	"github.com/kshedden/dstream/dstream"
)

type chunkMoment struct {

	// The data used to perform the analysis
	Data dstream.Reg

	// Sample size, mean and covariance, separately by y level, but
	// marginal over chunks.
	ny   []int
	mean [][]float64
	cov  [][]float64

	ntot   int // overall sample size
	nchunk int // number of chunks

	chunkmeans [][][]float64
	chunkn     [][]int

	// Marginal mean and covariance
	margmean []float64
	margcov  []float64

	log *log.Logger
}

func (cm *chunkMoment) MargCov() []float64 {
	return cm.margcov
}

func (cm *chunkMoment) MargMean() []float64 {
	return cm.margmean
}

func (cm *chunkMoment) YCov(y int) []float64 {
	return cm.cov[y]

}

func (cm *chunkMoment) YMean(y int) []float64 {
	return cm.mean[y]
}

// zero a stack of workspaces
func zero(x [][]float64) [][]float64 {
	x = x[0:cap(x)]
	for _, v := range x {
		for j, _ := range v {
			v[j] = 0
		}
	}
	return x[0:0]
}

// addBetweenCov adds the covariance of between-chunk means to the
// current value of cov (everything stratified by y level).
func (cm *chunkMoment) addBetweenCov() {

	p := cm.Data.NCov()
	pp := p * p
	wk := make([]float64, pp)

	for yi := 0; yi < len(cm.mean); yi++ {

		if cm.ny[yi] == 0 {
			continue
		}

		for i, _ := range wk {
			wk[i] = 0
		}

		for j, chunk := range cm.chunkmeans {
			for k1 := 0; k1 < p; k1++ {
				if len(chunk) <= yi {
					continue
				}
				u1 := chunk[yi][k1] - cm.mean[yi][k1]
				for k2 := 0; k2 <= k1; k2++ {
					u2 := chunk[yi][k2] - cm.mean[yi][k2]
					wk[k1*p+k2] += float64(cm.chunkn[j][yi]) * u1 * u2
				}
			}
		}

		reflect(wk, p)
		floats.Scale(1/float64(cm.ny[yi]), wk)
		floats.AddScaled(cm.cov[yi], 1, wk)
	}
}

// chunkMean calculates the means of the current chunk's data,
// stratifying by y-level.
func (cm *chunkMoment) chunkMean() ([][]float64, []int) {
	p := cm.Data.NCov()
	y := cm.Data.YData()
	var mn [][]float64
	var ny []int
	for j := 0; j < p; j++ {
		x := cm.Data.XData(j)
		for i := 0; i < len(y); i++ {
			yi := int(y[i])
			for len(mn) <= yi {
				if cap(mn) > yi {
					mn = mn[0 : yi+1]
				} else {
					mn = append(mn, make([]float64, p))
				}
				ny = append(ny, 0)
			}
			mn[yi][j] += x[i]
			if j == 0 {
				ny[yi]++
			}
		}
	}
	for i, v := range mn {
		if ny[i] > 0 {
			floats.Scale(1/float64(ny[i]), v)
		}
	}

	return mn, ny
}

// chunkCov calculates the covariances of the current chunk's data,
// stratifying by y-level.  The provided storage is used if possible.
func (cm *chunkMoment) chunkCov(cov [][]float64, mn [][]float64, ny []int) [][]float64 {
	p := cm.Data.NCov()
	pp := p * p
	cov = zero(cov)
	y := cm.Data.YData()
	for j1 := 0; j1 < p; j1++ {
		x1 := cm.Data.XData(j1)
		for j2 := 0; j2 <= j1; j2++ {
			x2 := cm.Data.XData(j2)
			for i := 0; i < len(y); i++ {
				yi := int(y[i])
				for len(cov) <= yi {
					if cap(cov) > yi {
						cov = cov[0 : yi+1]
					} else {
						cov = append(cov, make([]float64, pp))
					}
				}
				cov[yi][j1*p+j2] += (x1[i] - mn[yi][j1]) * (x2[i] - mn[yi][j2])
			}
		}
	}
	for i, v := range cov {
		reflect(v, p)
		if ny[i] > 0 {
			floats.Scale(1/float64(ny[i]), v)
		}
	}

	return cov
}

// walk cycles through the data and calculates chunk-wise summary
// statistics (mean and covariance by y level).
func (cm *chunkMoment) walk() {

	// Clear everything
	cm.Data.Reset()
	cm.ntot = 0
	cm.nchunk = 0
	cm.chunkn = cm.chunkn[0:0]
	cm.mean = cm.mean[0:0]
	cm.cov = cm.cov[0:0]
	cm.chunkmeans = cm.chunkmeans[0:0]
	cm.ny = cm.ny[0:0]

	p := cm.Data.NCov()
	pp := p * p
	var cov [][]float64 // stack of p^2 dimensional workspaces

	for cm.Data.Next() {

		y := cm.Data.YData()
		n := len(y)
		cm.ntot += n
		cm.nchunk++

		// Means for the current chunk (by y level)
		mn, ny := cm.chunkMean()

		// Covariances of current chunk (by y level)
		cov = cm.chunkCov(cov, mn, ny)

		// Update the counts
		for i, n := range ny {
			for len(cm.ny) <= i {
				cm.ny = append(cm.ny, 0)
			}
			cm.ny[i] += n
		}
		cm.chunkn = append(cm.chunkn, ny)

		// Update the means
		for i, v := range mn {
			for len(cm.mean) <= i {
				cm.mean = append(cm.mean, make([]float64, p))
			}
			floats.AddScaled(cm.mean[i], float64(ny[i]), v)
		}

		// Update the mean covariances
		for i, v := range cov {
			for len(cm.cov) <= i {
				cm.cov = append(cm.cov, make([]float64, pp))
			}
			floats.AddScaled(cm.cov[i], float64(ny[i]), v)
		}

		cm.chunkmeans = append(cm.chunkmeans, mn)
	}

	// Normalize the means
	for i, v := range cm.mean {
		floats.Scale(1/float64(cm.ny[i]), v)
	}

	// Normalize the mean covariances
	for i, v := range cm.cov {
		floats.Scale(1/float64(cm.ny[i]), v)
	}

	cm.addBetweenCov()

	if cm.log != nil {
		cm.log.Printf("%d records", cm.ntot)
		cm.log.Printf("%d data chunks", cm.nchunk)
		cm.log.Printf("Sample sizes per chunk:")
		for j, chunkn := range cm.chunkn {
			cm.log.Printf(fmt.Sprintf("Chunk %6d %v", j, chunkn))
		}
		cm.log.Printf("Mean by y level:")
		for j, v := range cm.mean {
			cm.log.Printf("%d %v", j, v)
		}
		cm.log.Printf("Covariance by y level:")
		for j, v := range cm.cov {
			cm.log.Printf("%d %v", j, v)
		}
	}
}

// getMargMean collapses over the y levels to get the marginal mean
func (cm *chunkMoment) getMargMean() {

	p := cm.Data.NCov()
	margmean := make([]float64, p)

	for yi, v := range cm.mean {
		for k, u := range v {
			margmean[k] += float64(cm.ny[yi]) * u
		}
	}
	floats.Scale(1/float64(cm.ntot), margmean)
	if cm.log != nil {
		cm.log.Printf("Marginal mean:")
		cm.log.Printf(fmt.Sprintf("%v", margmean))
	}

	cm.margmean = margmean
}

// getMargCov collapses over the y levels to get the marginal covariance matrix
func (cm *chunkMoment) getMargCov() {

	if cm.margmean == nil {
		cm.getMargMean()
	}

	p := cm.Data.NCov()
	pp := p * p

	// Mean within-y covariance
	wc := make([]float64, pp)
	for yi, v := range cm.cov {
		floats.AddScaled(wc, float64(cm.ny[yi]), v)
	}
	floats.Scale(1/float64(cm.ntot), wc)

	// Covariance of within-y means
	bc := make([]float64, pp)
	for yi, mn := range cm.mean {
		for k1 := 0; k1 < p; k1++ {
			u1 := mn[k1] - cm.margmean[k1]
			for k2 := 0; k2 <= k1; k2++ {
				u2 := mn[k2] - cm.margmean[k2]
				bc[k1*p+k2] += float64(cm.ny[yi]) * u1 * u2
			}
		}
	}
	reflect(bc, p)
	floats.Scale(1/float64(cm.ntot), bc)

	cm.margcov = make([]float64, pp)
	for i := 0; i < pp; i++ {
		cm.margcov[i] = bc[i] + wc[i]
	}

	if cm.log != nil {
		cm.log.Printf("Marginal covariance:")
		cm.log.Printf(fmt.Sprintf("%v", cm.margcov))
	}
}
