// Load in dependencies
var expect = require('chai').expect;
var fsUtils = require('./utils/fs');
var gruntUtils = require('./utils/grunt');

// Begin our tests
describe('A grunt `zip` task', function () {
  describe('zipping a single file', function () {
    gruntUtils.runTask('zip:single');

    it('matches the expected output', function () {
      fsUtils.assertCloseFiles('single_zip/file.zip', 150);
    });
  });

  describe('zipping multiple file', function () {
    gruntUtils.runTask('zip:multi');

    it('matches the expected output', function () {
      fsUtils.assertCloseFiles('multi_zip/file.zip', 150);
    });
  });

  describe('zipping a binary file (image)', function () {
    gruntUtils.runTask('zip:image');
    gruntUtils.runTask('unzip:test-zip-image');

    it('does not corrupt the file', function () {
      fsUtils.assertEqualFiles('image_zip/unzip/test_files/smile.gif');
    });
  });

  describe('zipping nested folders', function () {
    gruntUtils.runTask('zip:nested');
    gruntUtils.runTask('unzip:test-zip-nested');

    it('saves the nested files', function () {
      fsUtils.assertEqualFiles('nested_zip/unzip/test_files/nested/hello.js');
      fsUtils.assertEqualFiles('nested_zip/unzip/test_files/nested/world.txt');
      fsUtils.assertEqualFiles('nested_zip/unzip/test_files/nested/glyphicons-halflings.png');
      fsUtils.assertEqualFiles('nested_zip/unzip/test_files/nested/nested2/hello10.txt');
      fsUtils.assertEqualFiles('nested_zip/unzip/test_files/nested/nested2/hello20.js');
    });
  });

  describe('zipping files with a `router`', function () {
    gruntUtils.runTask('zip:router');
    gruntUtils.runTask('unzip:test-zip-router');

    it('routes the files', function () {
      fsUtils.assertEqualFiles('router_zip/unzip/hello.js');
      fsUtils.assertEqualFiles('router_zip/unzip/hello10.txt');
    });
  });

  describe('zipping files with a `cwd` parameter', function () {
    gruntUtils.runTask('zip:cwd');
    gruntUtils.runTask('unzip:test-zip-cwd');

    it('adjusts the filepaths', function () {
      fsUtils.assertEqualFiles('cwd_zip/unzip/hello.js');
      fsUtils.assertEqualFiles('cwd_zip/unzip/nested2/hello10.txt');
    });
  });

  describe('zipping dot files', function () {
    gruntUtils.runTask('zip:dot');
    gruntUtils.runTask('unzip:test-zip-dot');

    it('saves the dot files', function () {
      fsUtils.assertEqualFiles('dot_zip/unzip/test_files/dot/.test/hello.js');
      fsUtils.assertEqualFiles('dot_zip/unzip/test_files/dot/test/.examplerc');
    });
  });

  describe('zipping files with a router that skips files', function () {
    gruntUtils.runTask('zip:skip-files');
    gruntUtils.runTask('unzip:test-zip-skip-files');

    it('saves normal files', function () {
      fsUtils.assertEqualFiles('skip_files_zip/unzip/test_files/nested/hello.js');
    });

    it('does not save skipped files', function () {
      fsUtils.assertNoFile('skip_files_zip/unzip/test_files/nested/nested2/hello10.txt');
    });
  });
});

// If we are in `grunt>=0.3`, load a version specific test
var gruntInfo = require('grunt/package.json');
if (!gruntInfo.version.match(/^0.3.\d+$/)) {
  describe('A `zip` task with destination templating', function () {
    // 0.4 specific test for twolfson/grunt-zip#6
    gruntUtils.runTask('zip:actual/template_zip/<%= pkg.name %>.zip');
    fsUtils.exists( __dirname + '/actual/template_zip/grunt-zip.zip');

    it('generates the target file', function () {
      expect(this.fileExists).to.equal(true);
    });
  });
}
