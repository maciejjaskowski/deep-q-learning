// Load in dependencies
var fs = require('fs');
var expect = require('chai').expect;
var fsUtils = require('./utils/fs');
var gruntUtils = require('./utils/grunt');

// Begin our tests
describe('A grunt `unzip` task', function () {
  describe('unzipping a single file', function () {
    gruntUtils.runTask('unzip:single');

    it('generates a matching first file', function () {
      fsUtils.assertEqualFiles('single_unzip/a.js');
    });

    it('generates a matching second file', function () {
      fsUtils.assertEqualFiles('single_unzip/b.js');
    });
  });

  describe('unzipping a file with nested folders', function () {
    gruntUtils.runTask('unzip:nested');

    it('extracts all expected files', function () {
      fsUtils.assertEqualFiles('nested_unzip/bootstrap/css/bootstrap-responsive.css');
      fsUtils.assertEqualFiles('nested_unzip/bootstrap/css/bootstrap-responsive.min.css');
      fsUtils.assertEqualFiles('nested_unzip/bootstrap/css/bootstrap.css');
      fsUtils.assertEqualFiles('nested_unzip/bootstrap/css/bootstrap.min.css');
      fsUtils.assertEqualFiles('nested_unzip/bootstrap/img/glyphicons-halflings-white.png');
      fsUtils.assertEqualFiles('nested_unzip/bootstrap/img/glyphicons-halflings.png');
      fsUtils.assertEqualFiles('nested_unzip/bootstrap/js/bootstrap.js');
      fsUtils.assertEqualFiles('nested_unzip/bootstrap/js/bootstrap.min.js');
    });
  });

  describe('unzipping a file with a `router` parameter', function () {
    gruntUtils.runTask('unzip:router');

    it('maps the files to their new locations', function () {
      fsUtils.assertEqualFiles('router_unzip/bootstrap-responsive.css');
      fsUtils.assertEqualFiles('router_unzip/bootstrap-responsive.min.css');
      fsUtils.assertEqualFiles('router_unzip/bootstrap.css');
      fsUtils.assertEqualFiles('router_unzip/bootstrap.min.css');
      fsUtils.assertEqualFiles('router_unzip/glyphicons-halflings-white.png');
      fsUtils.assertEqualFiles('router_unzip/glyphicons-halflings.png');
      fsUtils.assertEqualFiles('router_unzip/bootstrap.js');
      fsUtils.assertEqualFiles('router_unzip/bootstrap.min.js');
    });
  });

  describe('unzipping a file with a `router` that skips files', function () {
    gruntUtils.runTask('unzip:skip-files');

    it('creates un-skipped files', function () {
      fsUtils.assertEqualFiles('skip_files_unzip/bootstrap/img/glyphicons-halflings-white.png');
      fsUtils.assertEqualFiles('skip_files_unzip/bootstrap/img/glyphicons-halflings.png');
      fsUtils.assertEqualFiles('skip_files_unzip/bootstrap/js/bootstrap.js');
      fsUtils.assertEqualFiles('skip_files_unzip/bootstrap/js/bootstrap.min.js');
    });

    it('does not create skipped files', function () {
      fsUtils.assertNoFile('skip_files_unzip/bootstrap/css/bootstrap-responsive.css');
      fsUtils.assertNoFile('skip_files_unzip/bootstrap/css/bootstrap-responsive.min.css');
      fsUtils.assertNoFile('skip_files_unzip/bootstrap/css/bootstrap.css');
      fsUtils.assertNoFile('skip_files_unzip/bootstrap/css/bootstrap.min.css');
    });
  });

  describe('unzipping a file with nested empty directories', function () {
    gruntUtils.runTask('unzip:empty');

    it('creates the nested empty directories', function () {
      var stats = fs.statSync('actual/empty/double_empty');
      expect(stats.isDirectory()).to.equal(true);
    });
  });

  describe('unzipping a file with permissions on its files', function () {
    gruntUtils.runTask('unzip:permissioned');

    it('preserves the permissions', function () {
      var stats = fs.statSync('actual/permissioned/permissioned-file');
      expect(stats.mode).to.equal(0100600);
    });
  });
});
