# Changelog

## v2.0.0:

- Added filters support:
  - Normal map Strength linear
  - Normal map Strength cubic *(contrast)*
- 2 new scaling algorithms:
  - Hamming *(PIL)*
  - Box *(PIL)* *(alternative to CV2 Inter Area)*
- Added conversion support
  - Quicker file format conversion *(e.g. PNG to WEBP, or compression)*
  - Advanced conversions:
    - MC texture old Continuum PBR to LabPBR 1.3
- Complete code rewrite and project restructure
- Various fixes and optimizations
- Plugin support *(right now only kind-off, but should work)*
- Better console user interface
- Added Unittests to assure everything will work during next releases
- Full, better, native CLI support

## Older

For older changelog the only source are [GitHub commits](https://github.com/MikiP98/MultiScaler-Plus/commits)

<br>

## Versioning methodology

### `{Major}.{Minor}.{Patch}`  

- **Major** &nbsp; --> &nbsp; big changes or entire new features  
- **Minor** &nbsp; --> &nbsp; current features improvements and added content  
- **Patch** &nbsp; --> &nbsp; bug fixes/typo fixes/markdown updates