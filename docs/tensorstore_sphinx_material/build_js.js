const esbuild = require('esbuild');
const path = require('path');

if (!process.env.HOME) {
  // When invoked by Bazel, HOME may be unset which can cause problems with
  // uv_os_homedir.
  process.env.HOME = process.cwd();
}

async function main() {
  const [, , outputPath, outputMapPath] = process.argv;
  const inputPath = path.join(__dirname, 'src/assets/javascripts/bundle.ts');
  await esbuild.build({
    entryPoints: [inputPath],
    target: 'es2015',
    outfile: outputPath,
    preserveSymlinks: true,
    bundle: true,
    sourcemap: true,
    minify: process.argv.includes('--optimize')
  });
}

main();
