// Renders SVG to PNG with a local Chrome (supports HTML/CSS inside
// <foreignObject>). Usage: node scripts/svg2png.js <input.svg> [output.png]
// [scale]; scale defaults to 2x the SVG size for papers / high-DPI screens.

const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');
const os = require('os');

const CHROME_CANDIDATES = [
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
  'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe',
  'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
  'C:/Program Files/Microsoft/Edge/Application/msedge.exe',
];

function findChrome() {
  for (const p of CHROME_CANDIDATES) if (fs.existsSync(p)) return p;
  throw new Error('Chrome or Edge not found; install one of them');
}

function parseSvgSize(svg) {
  // Prefer viewBox, then width/height
  const vb = svg.match(/viewBox\s*=\s*"([^"]+)"/i);
  if (vb) {
    const parts = vb[1].trim().split(/[\s,]+/).map(Number);
    if (parts.length === 4 && parts.every(Number.isFinite)) {
      return { width: parts[2], height: parts[3] };
    }
  }
  const w = svg.match(/<svg[^>]*\swidth\s*=\s*"([\d.]+)/i);
  const h = svg.match(/<svg[^>]*\sheight\s*=\s*"([\d.]+)/i);
  if (w && h) return { width: parseFloat(w[1]), height: parseFloat(h[1]) };
  throw new Error('Could not parse SVG size');
}

function main() {
  const [, , inputArg, outputArg, scaleArg] = process.argv;
  if (!inputArg) {
    console.error('Usage: node svg2png.js <input.svg> [output.png] [scale]');
    process.exit(1);
  }
  const inputAbs = path.resolve(inputArg);
  if (!fs.existsSync(inputAbs)) throw new Error('Input file not found: ' + inputAbs);

  const outputAbs = path.resolve(
    outputArg || inputAbs.replace(/\.svg$/i, '.png')
  );
  const scale = Math.max(1, parseFloat(scaleArg || '2'));

  const svg = fs.readFileSync(inputAbs, 'utf8');
  const { width, height } = parseSvgSize(svg);
  const outW = Math.round(width * scale);
  const outH = Math.round(height * scale);

  // Embed the SVG in HTML, sized to the viewport; body has no padding so the
  // screenshot is borderless
  const html =
    '<!doctype html><html><head><meta charset="utf-8">' +
    '<style>html,body{margin:0;padding:0;background:#fff;}' +
    'svg{display:block;width:' + outW + 'px;height:' + outH + 'px;}' +
    '</style></head><body>' + svg + '</body></html>';

  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'svg2png-'));
  const htmlPath = path.join(tmpDir, 'index.html');
  fs.writeFileSync(htmlPath, html, 'utf8');

  const chrome = findChrome();
  const fileUrl = 'file:///' + htmlPath.replace(/\\/g, '/');

  const args = [
    '--headless=new',
    '--disable-gpu',
    '--hide-scrollbars',
    '--no-sandbox',
    '--default-background-color=00000000', // transparent; body decides
    '--window-size=' + outW + ',' + outH,
    '--screenshot=' + outputAbs,
    fileUrl,
  ];

  console.log('[svg2png] chrome:', chrome);
  console.log('[svg2png] size  :', outW + 'x' + outH, '(scale=' + scale + ')');
  console.log('[svg2png] out   :', outputAbs);

  execFileSync(chrome, args, { stdio: 'inherit' });

  // Clean up the temp directory
  try { fs.rmSync(tmpDir, { recursive: true, force: true }); } catch (e) {}

  const stat = fs.statSync(outputAbs);
  console.log('[svg2png] done , size =', stat.size, 'bytes');
}

main();
