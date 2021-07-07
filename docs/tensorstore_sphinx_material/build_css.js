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

function normalizeSvg(code) {
  return code.replace(/<\?xml[^>]*>/, '')
      .replace(/<!DOCTYPE[^>]*>/, '')
      .replace(/'/g, '%22')
      .replace(/"/g, '\'')
      .replace(/\s+/g, ' ')
      .trim();
}

async function convertIconPaths(cssContent, iconAliases) {
  const pattern = new RegExp(
      'svg-load\\s*\\(\\s*([\'"])(' + Object.keys(iconAliases).join('|') +
          ')/([/a-z0-9\\-_]+\\.svg)\\1\\s*\\)',
      'g');
  const requiredPaths = new Map();
  const cssContentString = cssContent.toString();
  cssContentString.replace(pattern, (match, quote, alias, name) => {
    requiredPaths.set(require.resolve(iconAliases[alias] + name), null);
  });
  await Promise.all(Array.from(requiredPaths.keys(), async (p) => {
    requiredPaths.set(p, await fs.readFile(p, 'utf-8'));
  }));
  return cssContentString.replace(pattern, (match, quote, alias, name) => {
    return 'url(' +
        JSON.stringify(
            'data:image/svg+xml;charset=utf-8,' +
            normalizeSvg(requiredPaths.get(
                require.resolve(iconAliases[alias] + name)))) +
        ')';
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
  css0 = await convertIconPaths(css0, iconAliases);
  const {css: css1, map: map1} =
      await postcss([
        require('autoprefixer'),
        ...process.argv.includes('--optimize') ?
            [require('cssnano')({preset: 'default'})] :
            []
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
