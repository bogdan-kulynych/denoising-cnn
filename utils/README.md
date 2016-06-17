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
