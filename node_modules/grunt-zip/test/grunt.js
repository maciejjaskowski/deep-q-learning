var path = require('path');

module.exports = function (grunt) {

  // TODO:
  // // Add in 0.4 specific tests
  // var _ = grunt.util._;
  // var zipConfig = grunt.config.get('zip');
  // grunt.config.set('zip', _.extend(zipConfig, {
  //   'actual/template_zip/<%= pkg.name %>.zip': ['test_files/file.js']
  // }));

  // Project configuration.
  grunt.initConfig({
    // DEV: `pkg` is used for template test
    pkg: require('../package.json'),
    zip: {
      single: {
        src: ['test_files/file.js'],
        dest: 'actual/single_zip/file.zip'
      },
      multi: {
        src: ['test_files/file.js', 'test_files/file2.js'],
        dest: 'actual/multi_zip/file.zip'
      },
      nested: {
        src: 'test_files/nested/**/*',
        dest: 'actual/nested_zip/file.zip'
      },
      image: {
        src: 'test_files/smile.gif',
        dest: 'actual/image_zip/file.zip'
      },
      router: {
        src: ['test_files/nested/hello.js', 'test_files/nested/nested2/hello10.txt'],
        dest: 'actual/router_zip/file.zip',
        router: function (filepath) {
          var filename = path.basename(filepath);
          return filename;
        }
      },
      cwd: {
        src: ['test_files/nested/hello.js', 'test_files/nested/nested2/hello10.txt'],
        dest: 'actual/cwd_zip/file.zip',
        cwd: 'test_files/nested'
      },
      dot: {
        src: ['test_files/dot/.test/hello.js', 'test_files/dot/test/.examplerc'],
        dest: 'actual/dot_zip/file.zip',
        dot: true
      },
      'skip-files': {
        src: ['test_files/nested/hello.js', 'test_files/nested/nested2/hello10.txt'],
        dest: 'actual/skip_files_zip/file.zip',
        router: function (filepath) {
          // Skip over txt files
          return filepath.indexOf('.txt') === -1 ? filepath : null;
        }
      }
    },
    unzip: {
      single: {
        src: 'test_files/file.zip',
        dest: 'actual/single_unzip'
      },
      nested: {
        src: 'test_files/nested.zip',
        dest: 'actual/nested_unzip'
      },
      router: {
        src: 'test_files/nested.zip',
        dest: 'actual/router_unzip',
        router: function (filepath) {
          var filename = path.basename(filepath);
          return filename;
        }
      },
      'skip-files': {
        src: 'test_files/nested.zip',
        dest: 'actual/skip_files_unzip',
        router: function (filepath) {
          // Skip over css files
          return filepath.indexOf('.css') === -1 ? filepath : null;
        }
      },
      empty: {
        src: 'test_files/empty.zip',
        dest: 'actual/empty'
      },
      permissioned: {
        src: 'test_files/permissioned.zip',
        dest: 'actual/permissioned'
      },
      'test-zip-nested': {
        src: 'actual/nested_zip/file.zip',
        dest: 'actual/nested_zip/unzip'
      },
      'test-zip-image': {
        src: 'actual/image_zip/file.zip',
        dest: 'actual/image_zip/unzip'
      },
      'test-zip-router': {
        src: 'actual/router_zip/file.zip',
        dest: 'actual/router_zip/unzip'
      },
      'test-zip-cwd': {
        src: 'actual/cwd_zip/file.zip',
        dest: 'actual/cwd_zip/unzip'
      },
      'test-zip-dot': {
        src: 'actual/dot_zip/file.zip',
        dest: 'actual/dot_zip/unzip'
      },
      'test-zip-skip-files': {
        src: 'actual/skip_files_zip/file.zip',
        dest: 'actual/skip_files_zip/unzip'
      },
    }
  });

  // Load local tasks.
  grunt.loadTasks('../tasks');

  // Load grunt contrib clean (chdir for 0.4)
  process.chdir('..');
  grunt.loadNpmTasks('grunt-contrib-clean');
  process.chdir(__dirname);
};
