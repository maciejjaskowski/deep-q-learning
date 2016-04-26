module.exports = function (grunt) {
  // Define a zip
  grunt.initConfig({
    zip: {
      'location/to/zip/files.zip': ['file/to/zip.js', 'another/file.css']
    },
    unzip: {
      'location/to/extract/files/': 'file/to/extract.zip'
    }
  });

  // Load in `grunt-zip`
  grunt.loadTasks('../tasks');
};
