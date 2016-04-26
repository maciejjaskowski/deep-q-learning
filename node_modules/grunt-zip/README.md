# grunt-zip [![Build status](https://travis-ci.org/twolfson/grunt-zip.png?branch=master)](https://travis-ci.org/twolfson/grunt-zip)

Zip and unzip files via a [grunt][] plugin

This was created for dependency management via [`grunt-curl`][] and [`grunt-zip`][] as a low-tech alternative to `bower` and similar solutions.

http://twolfson.com/2014-01-19-low-tech-dependency-management-via-grunt-tasks

[grunt]: http://gruntjs.com/
[`grunt-curl`]: https://github.com/twolfson/grunt-curl
[`grunt-zip`]: https://github.com/twolfson/grunt-zip

**Features**

- Zip and unzip support
- Per-file renaming/routing via `router` option
- File skipping via `router` option
- Preservation of UNIX file permissions during extraction

## Getting Started
`grunt-zip` can be installed via npm: `npm install grunt-zip`

`grunt-zip` provides 2 grunt tasks: `zip` and `unzip`

### zip
Compress files into a `zip` file

```js
// Inside your Gruntfile.js
module.exports = function (grunt) {
  // Define a zip task
  grunt.initConfig({
    zip: {
      'location/to/zip/files.zip': ['file/to/zip.js', 'another/file.css']
    }
  });

  // Load in `grunt-zip`
  grunt.loadNpmTasks('grunt-zip');
};
```

Now, we can run our task:

```bash
$ grunt zip
Running "zip:location/to/zip/files.zip" (zip) task
File "location/to/zip/files.zip" created.

Done, without errors.
```

### unzip
Extract files from a `zip` file

```js
// Inside your Gruntfile.js
module.exports = function (grunt) {
  // Define an unzip task
  grunt.initConfig({
    unzip: {
      'location/to/extract/files/': 'file/to/extract.zip'
    }
  });

  // Load in `grunt-zip`
  grunt.loadNpmTasks('grunt-zip');
};
```

Now, we can run our task:

```bash
$ grunt unzip
Running "unzip:location/to/extract/files/" (unzip) task
Created "location/to/extract/files/" directory

Done, without errors.
```

## Documentation
### zip
#### Short format
The short format relies on [`grunt's` support of `{dest: src}`][grunt-short-format]

[grunt-short-format]: http://gruntjs.com/configuring-tasks#older-formats

```js
zip: {
  'location/to/zip/files.zip': ['file/to/zip.js', 'another/file.css']
}
```

This format is suggested only if you don't need to run `zip` tasks separately

```bash
grunt zip
```

If you want to run this task standalone, it must be executed via:

```bash
grunt zip:dest
# grunt zip:location/to/zip/files.zip
```

#### Long format
```js
zip: {
  'long-format': {
    src: ['file/to/zip.js', 'another/file.css'],
    dest: 'location/to/zip/files.zip'
  }
}
```

#### Using `cwd`
The `cwd` option sets the base path for the zipped files

```js
zip: {
  'using-cwd': {
    cwd: 'nested/'
    // Files will zip to 'hello.js' and 'world.js'
    src: ['nested/hello.js', 'nested/world.js'],
    dest: 'location/to/zip/files.zip'
  }
}
```

#### Using `router`
The `router` option allows for adjust file paths on a per-file basis. This *cannot* be used with `cwd` since there are ordering conflicts.

```js
// This example requires using node's `path` module
var path = require('path');

// Inside grunt config
zip: {
  'using-router': {
    // `router` receives the path from grunt (e.g. js/main.js)
    // The path it returns is what the file contents are saved as (e.g. all/main.js)
    router: function (filepath) {
      // Route each file to all/{{filename}}
      var filename = path.basename(filepath);
      return 'all/' + filename;
    },

    // Files will zip to 'main.js' and 'main.css'
    src: ['js/main.js', 'css/main.css'],
    dest: 'files.zip'
  }
}
```

#### Remaining options
We allow for specifying the `DEFLATE` comrpession algorithm, base64 encoding, and including `dotfiles` (e.g. `.travis.yml`) via the following options:

```js
zip: {
  'using-delate': {
    src: ['file.js'],
    dest: 'files.zip',
    compression: 'DEFLATE'
  },
  'encode-base64': {
    src: ['file.js'],
    dest: 'files.zip',
    base64: true
  },
  'including-dotfiles': {
    src: ['file.js'],
    dest: 'files.zip',
    dot: true
  }
}
```

### unzip
#### Short format
As with `zip`, we support the `{dest: src}` format. Additionally, it has the same drawbacks of being difficult to run standalone.

```js
unzip: {
  'location/to/extract/files': 'file/to/extract.zip'
}
```

#### Long format
```js
unzip: {
  'long-format': {
    // Note: If you provide multiple src files, they will all be extracted to the same folder.
    // This is not well-tested behavior so use at your own risk.
    src: 'file/to/extract.zip',
    dest: 'location/to/extract/files'
  }
}
```

#### Using `router`
During extraction, we can dynamically change the filepaths of the `zip's` contents via the `router` option.

```js
// This example requires using node's `path` module
var path = require('path');

// Inside grunt config
unzip: {
  'using-router': {
    // `router` receives the path that was used during zipping (e.g. css/bootstrap.css)
    // The path it returns is where the file contents will be written to (e.g. dist/bootstrap.css)
    router: function (filepath) {
      // Route each file to dist/{{filename}}
      var filename = path.basename(filepath);
      return 'dist/' + filename;
    },

    // Collects all nested files in same directory
    // css/bootstrap.css -> bootstrap.css, js/bootstrap.js -> bootstrap.js
    src: 'bootstrap.zip',
    dest: 'bootstrap/'
  }
}
```

#### Remaining options
With the following options we can disable the CRC32 check or decode from base64 encoding:

```js
zip: {
  'skip-crc32-check': {
    src: 'bootstrap.zip',
    dest: 'bootstrap/',
    checkCRC32: false
  },
  'decode-base64': {
    src: ['file.js'],
    dest: 'files.zip',
    base64: true
  }
}
```

## Examples
### Skipping files during `zip`
`zip's router` allows for returning `null` to skip over a file.

```js
zip: {
  'skip-files': {
    // Skip over non-js files
    router: function (filepath) {
      // Grab the extension
      var extname = path.extname(filepath);

      // If the file is a .js, add it to the zip
      if (extname === '.js') {
        return filepath;
      } else {
      // Otherwise, skip it
        return null;
      }
    },

    src: ['js/main.js', 'css/main.css'],
    dest: 'js-only.zip'
  }
}
```

### Skipping files during `unzip`
As with `zip`, `unzip` supports skipping extracting of files by returning `null` in `router`.

```js
unzip: {
  'skip-files': {
    // Skip over non-css files
    router: function (filepath) {
      // Grab the extension
      var extname = path.extname(filepath);

      // If the file is a .css, extract it
      if (extname === '.css') {
        return filepath;
      } else {
      // Otherwise, skip it
        return null;
      }
    },

    src: ['bootstrap.css'],
    dest: 'bootstrap-css/'
  }
}
```

## Contributing
In lieu of a formal styleguide, take care to maintain the existing coding style. Add unit tests for any new or changed functionality. Lint your code using [grunt][grunt] and test via `npm test`.

## Donating
Support this project and [others by twolfson][gittip] via [gittip][].

[![Support via Gittip][gittip-badge]][gittip]

[gittip-badge]: https://rawgithub.com/twolfson/gittip-badge/master/dist/gittip.png
[gittip]: https://www.gittip.com/twolfson/

## License
Copyright (c) 2013 Todd Wolfson
Licensed under the MIT license.
