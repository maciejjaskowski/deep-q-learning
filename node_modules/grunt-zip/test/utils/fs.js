// Load in dependencis
var assert = require('assert');
var fs = require('fs');
var _ = require('underscore.string');

// Define common helpers
exports.exists = function (path) {
  before(function loadFile (done) {
    var that = this;
    fs.exists(path, function (exists) {
      that.fileExists = exists;
      done();
    });
  });
  after(function cleanupFile () {
    delete this.fileExists;
  });
};

// Define test assertions
exports.assertEqualFiles = function (filename) {
  // Read in content
  var expectedContent = fs.readFileSync('expected/' + filename, 'binary');
  var actualContent = fs.readFileSync('actual/' + filename, 'binary');

  // Assert that the content is *exactly* the same
  assert.strictEqual(actualContent, expectedContent, filename + ' does not have the same content in `expected` as `actual`');
};

// Assert two files are close enough
// ANTI-PATTERN: 3 specifically ordered/non-modular parameters =(
exports.assertCloseFiles = function (filename, distance) {
  // Read in the content
  var expectedContent = fs.readFileSync('expected/' + filename, 'binary');
  var actualContent = fs.readFileSync('actual/' + filename, 'binary');

  // Calculate the difference in bits (accounts for random bits)
  var difference = _.levenshtein(expectedContent, actualContent);

  // Assert that we are under our threshold
  var underThreshold = difference <= distance;
  assert.ok(underThreshold, 'Bitwise difference of zip files "' + difference + '" should be under ' + distance + ' (' + filename + ')');
};

// Assert file does not exist
exports.assertNoFile = function (filename) {
  try {
    // Attempt to grab stats on the would-be location
    fs.statSync('actual/' + filename);

    // Fail since there are statistics (should be file not found)
    assert.fail('File "' + filename + '" was found when it was expected to not exist');
  } catch (e) {
    // Verify the error is ENOENT
    assert.strictEqual(e.code, 'ENOENT', filename + ' exists');
  }
};
