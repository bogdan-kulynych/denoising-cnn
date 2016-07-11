# Utils

## Image filters

- [ ] Small affine transformations
- [ ] JPEG compression
- [ ] Sharpen
- [X] Gaussian blur
- [X] Gaussian noise
- [X] Uniform noise
- [X] Instagram color filters
- [X] Vignette

### Workflow

#### Write a filter
Add filter class in `utils/tests/filters.py`. Class should have `apply` method that takes a PIL image as input and returns a new PIL image with filter applied. You can use `_imagemagick` utility function to call imagemagick `convert` and get converted PIL image back.

#### Test a filter
1. Add filter parameters in `utils/tests/test_filters.py`. Run tests with `py.test` from the project root
2. Apply filter and see the result in `notebooks/filters.ipynb`. To use notebooks, start notebook server with `jupyter notebook`


## Data generator

### `utils.preprocessing.FilterImageDataGenerator`
Generates mini-batches of images with filter transformations applied, labels being transformation IDs.

#### Usage

```python
data_generator = FilterImageDataGenerator()
# Fitting generator is required for centering images, which is on by
# default.
data_generator.fit(data)
# List of filters that will be applied. Noop is required if original image
# needs to appear in the mini-batches
filters = [Noop(), UniformNoise(), Lomo()]
# `flow` creates an infinite (cyclic) iterator over the mini-batches
batch_iterator = data_generator.flow(data, filters, batch_size=batch_size)
for b in range(math.ceil(N / batch_size)):
    batch_X, batch_y = next(batch_iterator)
    # ...
```
