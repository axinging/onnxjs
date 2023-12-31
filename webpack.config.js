const util = require('util');
const path = require('path');
const webpack = require('webpack');
// const HardSourceWebpackPlugin = require('hard-source-webpack-plugin');

module.exports = (env, argv) => {
  const bundleMode = argv['mode'] || 'prod';  // 'prod'|'dev'|'perf'|undefined;

  const config = {
    resolve: {extensions: ['.ts', '.js']},
    plugins: [new webpack.WatchIgnorePlugin({paths:[/\.js$/, /\.d\.ts$/]})],
    //plugins: new webpack.WatchIgnorePlugin ({paths: [ 'src/test' ]}),
    module: {rules: [{test: /\.tsx?$/, loader: 'ts-loader'}]},
    //node: {fs: 'empty'}
  };

  if (bundleMode === 'perf' || bundleMode === 'dev') {
    //config.entry = path.resolve(__dirname, 'test/test-main.ts');
    config.entry = path.resolve(__dirname, 'lib/api/index.ts');
  } else {
    config.entry = path.resolve(__dirname, 'lib/api/index.ts');
  }
  console.log("bundleMode" + bundleMode);
  if (bundleMode === 'perf') {
    config.output = {path: path.resolve(__dirname, 'test'), filename: 'onnx.perf.js', libraryTarget: 'umd'};
  } else if (bundleMode === 'development') {
    config.output = {path: path.resolve(__dirname, 'test'), filename: 'onnx.dev.js', libraryTarget: 'umd', globalObject: 'this'};
  } else {
    config.output = {path: path.resolve(__dirname, 'dist'), filename: 'onnx.min.js', libraryTarget: 'umd',globalObject: 'this'};
  }

  if (bundleMode === 'prod') {
    config.mode = 'production';
    config.devtool = 'source-map';
  } else if (bundleMode === 'perf') {
    config.mode = 'production';
    config.devtool = '';
  } else {
    config.mode = 'development';
    config.devtool = 'inline-source-map';
    //config.plugins.push(new HardSourceWebpackPlugin());
  }

  return config;
};
