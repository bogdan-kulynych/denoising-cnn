# Utils

## Image filters

- [ ] Small affine transformations
- [ ] JPEG compression
- [X] Gaussian blur
- [ ] Sharpen
- [X] Gaussian noise
- [X] Uniform noise
- [ ] Instagram color filters
- [X] Vignette

### Workflow

How to test a filter:

1. Add filter parameters in `utils/tests/test_filters.py`. Run tests with `py.test` from the project root
2. Apply filter and see the result in `notebooks/filters.ipynb`. To use notebooks, start notebook server with `jupyter notebook`
