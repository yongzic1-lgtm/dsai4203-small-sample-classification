# 20261022-194211_r426_abc_attribution

- author: TODO_ID_1
- kind: analysis

## Hypothesis (written before running)

Written before running. The conditional public difference r201 - S003 is roughly flat across flip_y levels (-4e-4 to -6e-4), so the change from r936's 0.99884 to 0.99868 is mostly Monte-Carlo noise of r936's 125 accepted subsets (MC SE ~8e-5): r936's design with all 512 subsets per replica gives ~0.9987, and changing the weights from R156 to R922's flat prior moves the prediction by < 5e-5. Same for r209 (shift < 3e-5).

## Result

(see summary.json)

## Conclusion (write after reading the results)

Supported. The conditional public difference r201 - S003 is -4.3e-4 at f = 0 and -5.1e-4 to -6.1e-4 at every level from 0.001 to 0.03 (cluster SE 2.5-8e-5); r209 - S003 is -8e-5 at f = 0 and -1.2e-4 to -1.5e-4 elsewhere. r936's design gives r201 0.99882 +- 7.3e-5 (obs + mean difference; r936 reported the raw mean 0.99884) on 125 accepted subsets; the same design with all 512 subsets per replica gives 0.99869 +- 2.7e-5, R156 weights 0.99869, R922 flat prior truncated at 0.01 0.99869, and the full grid to 0.03 0.99868 +- 2.4e-5. So the 1.4e-4 drop from r936 to r425 is Monte-Carlo noise of r936's small accepted sample, not the grid extension (weights move the prediction by < 1e-5). r209: 0.99911 (r936 design) -> 0.99910 in every 512-subset design. (r2_noise)
