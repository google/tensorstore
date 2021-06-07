const postcss = require('postcss');
const sass = require('sass');
const {promisify} = require('util');

const fs = require('fs').promises;
const path = require('path');

if (!process.env.HOME) {
  // When invoked by Bazel, HOME may be unset which can cause problems with
  // uv_os_homedir.
  process.env.HOME = process.cwd();
}

function resolvePackageDir(package) {
  return path.dirname(require.resolve(package + '/package.json'));
}

function convertIconPaths(cssContent, iconAliases) {
  const pattern = new RegExp(
      '([\'"])(' + Object.keys(iconAliases).join('|') +
          ')/([/a-z0-9\\-_]+\\.svg)\\1',
      'g');
  return cssContent.toString().replace(pattern, (match, quote, alias, name) => {
    return require.resolve(iconAliases[alias] + name);
  });
}

async function transformStyle(
    iconAliases, inputPath, outputPath, outputMapPath) {
  const result = sass.renderSync({
    file: inputPath,
    outFile: outputPath,
    includePaths: [
      resolvePackageDir('material-design-color'),
      resolvePackageDir('material-shadows'),
    ],
    sourceMap: true,
    sourceMapContents: true
  });
  let {css: css0, map: map0} = result;
  css0 = convertIconPaths(css0, iconAliases);
  const {css: css1, map: map1} =
      await postcss([
        require('autoprefixer'), require('postcss-inline-svg')({
          encode: false,
        }),
        ...process.argv.includes('--optimize') ? [require('cssnano')] : []
      ]).process(css0, {
        from: inputPath,
        map: {
          prev: `${map0}`,
          inline: false,
        },
      });
  let missingSvgs = false;
  for (const match of css1.matchAll(/svg-load\([^\)]*\)/g)) {
    console.log('Failed to inline: ' + match[0]);
    missingSvgs = true;
  }
  if (missingSvgs) {
    process.exit(1);
  }
  await fs.writeFile(
      outputPath,
      css1.replace(
          /(sourceMappingURL=)(.*)/, `$1${path.basename(outputPath)}.map\n`));
  await fs.writeFile(outputMapPath, map1.toString());
}

async function main() {
  const [, , iconAliasesStr, inputPath1, inputPath2, outputPath1, outputMapPath1, outputPath2, outputMapPath2] =
      process.argv;
  const iconAliases =
      Object.fromEntries(iconAliasesStr.split(':').map(x => x.split('=')));
  await transformStyle(iconAliases, inputPath1, outputPath1, outputMapPath1);
  await transformStyle(iconAliases, inputPath2, outputPath2, outputMapPath2);
}

main();
