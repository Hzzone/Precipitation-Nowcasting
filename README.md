### Introduction
This repo has implemented a pytorch-based encoder-forecaster model with TrajGRU to do precipitation nowcasting. For more information about TrajGRU, please refer to [HKO-7](https://github.com/sxjscience/HKO-7).

### Demo
![](demo.gif)

### Performance
The performance on HKO-7 dataset is below.

<table>
	<tbody>
		<tr>
			<td colspan="5" align="center">CSI</td>
			<td colspan="5" align="center">HSS</td>
			<td align="center">Balanced MSE</td>
			<td align="center">Balanced MAE</td>
		</tr>
		<tr>
			<td  align="center"><img src="https://latex.codecogs.com/gif.latex?r&space;\geq&space;0.5" title="r \geq 0.5" /></td>
			<td align="center"><img src="https://latex.codecogs.com/gif.latex?r&space;\geq&space;2" title="r \geq 2" /></td>
			<td align="center"><img src="https://latex.codecogs.com/gif.latex?r&space;\geq&space;5" title="r \geq 5" /></td>
			<td align="center"><img src="https://latex.codecogs.com/gif.latex?r&space;\geq&space;10" title="r \geq 10" /></td>
			<td align="center"><img src="https://latex.codecogs.com/gif.latex?r&space;\geq&space;30" title="r \geq 30" /></td>
			<td align="center"><img src="https://latex.codecogs.com/gif.latex?r&space;\geq&space;0.5" title="r \geq 0.5" /></td>
			<td align="center"><img src="https://latex.codecogs.com/gif.latex?r&space;\geq&space;2" title="r \geq 2" /></td>
			<td align="center"><img src="https://latex.codecogs.com/gif.latex?r&space;\geq&space;5" title="r \geq 5" /></td>
			<td align="center"><img src="https://latex.codecogs.com/gif.latex?r&space;\geq&space;10" title="r \geq 10" /></td>
			<td align="center"><img src="https://latex.codecogs.com/gif.latex?r&space;\geq&space;30" title="r \geq 30" /></td>
			<td align="center"></td>
			<td align="center"></td>
		</tr>
		<tr>
			<td align="center">0.5496</td>
			<td align="center">0.4772</td>
			<td align="center">0.3774</td>
			<td align="center">0.2863</td>
			<td align="center">0.1794</td>
			<td align="center">0.6713</td>
			<td align="center">0.6150</td>
			<td align="center">0.5226</td>
			<td align="center">0.4253</td>
			<td align="center">0.2919</td>
			<td align="center">5860.97</td>
			<td align="center">15062.46</td>
		</tr>
	</tbody>
</table>

### Download

[Dropbox Pretrained Model](https://www.dropbox.com/sh/i5goltdq83dmkvc/AABBe5wTuEQF5j3VSMszVQSaa?dl=0)

