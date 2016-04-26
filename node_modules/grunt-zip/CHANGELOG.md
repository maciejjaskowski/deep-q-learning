# grunt-zip changelog
0.17.1 - Repaired test status and adjusted node version support

0.17.0 - Added support to extract UNIX file permissions

0.16.2 - Moved Travis CI to more restrictive `npm` upgrade to fix `node@0.8`

0.16.1 - Rewrote docs to make them more understandable

0.16.0 - Removed legacy filtering of leaf nodes. By @michaelsantiago

0.15.0 - Upgraded to `jszip@2.2.2`, fixes `DEFLATE` issue in #25

0.14.0 - Moved test suite to `mocha` to make standalone tests easier

0.13.0 - Added verbose output and updated unzip test for no src files. Fixes #24

0.12.0 - Robustified unzipping empty directories logic. Fixes #21 for OSX

0.11.0 - Fixed bug with unzipping empty directories. Fixes #21

0.10.2 - Added Travis CI

0.10.1 - Upgraded devDependencies from grunt@0.3 to grunt@0.4

0.10.0 - Fixed Windows -> Linux zip compatibility bug. Via #20

0.9.1 - Added assertion against `cwd` and `router` in same config

Before 0.9.1 - See `git log`
