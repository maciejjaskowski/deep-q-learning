module.exports = function (grunt) {
  // Load in legacy config
  require('./grunt')(grunt);

  // Add in 0.4 specific tests
  var _ = grunt.util._;
  var zipConfig = grunt.config.get('zip');
  grunt.config.set('zip', _.extend(zipConfig, {
    'actual/template_zip/<%= pkg.name %>.zip': ['test_files/file.js']
  }));
};
