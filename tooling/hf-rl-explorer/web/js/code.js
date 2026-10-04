// A small syntax highlighter for the file viewer: comments, strings, keywords, numbers, keys and diff lines, per line,
// with just enough state for the constructs that span lines (block comments, triple quotes, heredocs). It only adds
// spans around escaped text, so nothing in a file can become markup.
import { esc } from "./util.js";

const KW = {
  sh: "if then else elif fi for while until do done case esac in function return local export readonly set unset shift exit source alias declare trap eval exec break continue select time",
  py: "False None True and as assert async await break class continue def del elif else except finally for from global if import in is lambda nonlocal not or pass raise return try while with yield match case self print",
  js: "async await break case catch class const continue debugger default delete do else export extends false finally for from function if import in instanceof let new null of return static super switch this throw true try typeof undefined var void while with yield",
  go: "break case chan const continue default defer else fallthrough for func go goto if import interface map package range return select struct switch type var nil true false",
  rs: "as async await break const continue crate dyn else enum extern false fn for if impl in let loop match mod move mut pub ref return self Self static struct super trait true type unsafe use where while",
  c: "auto break case char const continue default do double else enum extern float for goto if inline int long register restrict return short signed sizeof static struct switch typedef union unsigned void volatile while bool true false NULL nullptr class public private protected virtual template typename namespace using new delete this throw try catch include define ifdef ifndef endif",
  java: "abstract boolean break byte case catch char class const continue default do double else enum extends final finally float for if implements import instanceof int interface long native new null package private protected public return short static super switch synchronized this throw throws transient true false try void volatile while var val fun",
  rb: "alias and begin break case class def defined do else elsif end ensure false for if in module next nil not or redo rescue retry return self super then true undef unless until when while yield require",
  sql: "select from where and or not insert into values update set delete create table drop alter index join left right inner outer on group by order having limit offset as distinct union all null is in like between case when then else end primary key foreign references",
  docker: "FROM RUN CMD LABEL MAINTAINER EXPOSE ENV ADD COPY ENTRYPOINT VOLUME USER WORKDIR ARG ONBUILD STOPSIGNAL HEALTHCHECK SHELL AS",
};
// case-insensitive keyword lists spell each letter both ways, so every rule can share one alternation without flags
const anyCase = (w) => w.replace(/[a-zA-Z]/g, (c) => `[${c.toLowerCase()}${c.toUpperCase()}]`);
const words = (s, caseless = false) => new RegExp(`\\b(?:${s.split(" ").map(caseless ? anyCase : (w) => w).join("|")})\\b`);

const NUM = /\b(?:0x[\da-fA-F]+|\d+(?:\.\d+)?(?:e[+-]?\d+)?)\b/;
const DQ = /"(?:[^"\\]|\\.)*"/, SQ = /'(?:[^'\\]|\\.)*'/, BQ = /`(?:[^`\\]|\\.)*`/;

// Each language: ordered [class, regex] rules (the first that matches at a position wins), plus openers for tokens
// that can run past the end of the line: [class, start regex, end regex].
const LANGS = {
  sh: { rules: [["c", /#.*$/], ["s", DQ], ["s", SQ], ["v", /\$\{[^}]*\}|\$[\w@#?*!$-]/], ["k", words(KW.sh)], ["n", NUM]],
        multi: [["s", /<<-?\s*['"]?(\w+)['"]?.*$/, "heredoc"]] },
  py: { rules: [["c", /#.*$/], ["m", /^\s*@[\w.]+/], ["s", /[rbfu]{0,2}"(?:[^"\\]|\\.)*"/], ["s", /[rbfu]{0,2}'(?:[^'\\]|\\.)*'/], ["k", words(KW.py)], ["n", NUM]],
        multi: [["s", /[rbfu]{0,2}"""/, /"""/], ["s", /[rbfu]{0,2}'''/, /'''/]] },
  js: { rules: [["c", /\/\/.*$/], ["s", DQ], ["s", SQ], ["s", BQ], ["k", words(KW.js)], ["n", NUM]], multi: [["c", /\/\*/, /\*\//]] },
  go: { rules: [["c", /\/\/.*$/], ["s", DQ], ["s", BQ], ["s", SQ], ["k", words(KW.go)], ["n", NUM]], multi: [["c", /\/\*/, /\*\//]] },
  rs: { rules: [["c", /\/\/.*$/], ["s", DQ], ["m", /#!?\[[^\]]*\]/], ["k", words(KW.rs)], ["n", NUM]], multi: [["c", /\/\*/, /\*\//]] },
  c: { rules: [["c", /\/\/.*$/], ["m", /^\s*#\s*\w+/], ["s", DQ], ["s", SQ], ["k", words(KW.c)], ["n", NUM]], multi: [["c", /\/\*/, /\*\//]] },
  java: { rules: [["c", /\/\/.*$/], ["m", /@\w+/], ["s", DQ], ["s", SQ], ["k", words(KW.java)], ["n", NUM]], multi: [["c", /\/\*/, /\*\//]] },
  rb: { rules: [["c", /#.*$/], ["s", DQ], ["s", SQ], ["v", /:[a-zA-Z_]\w*/], ["k", words(KW.rb)], ["n", NUM]] },
  sql: { rules: [["c", /--.*$/], ["s", SQ], ["s", DQ], ["k", words(KW.sql, true)], ["n", NUM]], multi: [["c", /\/\*/, /\*\//]] },
  toml: { rules: [["c", /#.*$/], ["h", /^\s*\[\[?[^\]]*\]\]?/], ["p", /^\s*[\w.-]+(?=\s*=)/], ["p", /^\s*"[^"]*"(?=\s*=)/], ["s", DQ], ["s", SQ], ["k", /\b(?:true|false)\b/], ["n", NUM]],
          multi: [["s", /"""/, /"""/], ["s", /'''/, /'''/]] },
  yaml: { rules: [["c", /(?:^|\s)#.*$/], ["m", /^---$|^\.\.\.$/], ["p", /^\s*-?\s*[\w.\-/ ]+(?=\s*:(?:\s|$))/], ["s", DQ], ["s", SQ], ["k", /\b(?:true|false|null|yes|no)\b/], ["n", NUM]] },
  json: { rules: [["p", /"(?:[^"\\]|\\.)*"(?=\s*:)/], ["s", DQ], ["k", /\b(?:true|false|null)\b/], ["n", /-?\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b/]] },
  docker: { rules: [["c", /^\s*#.*$/], ["k", new RegExp(`^\\s*(?:${KW.docker.split(" ").map(anyCase).join("|")})\\b`)], ["s", DQ], ["s", SQ], ["v", /\$\{[^}]*\}|\$\w+/], ["n", NUM]] },
  ini: { rules: [["c", /^\s*[#;].*$/], ["h", /^\s*\[[^\]]*\]/], ["p", /^\s*[\w.-]+(?=\s*[=:])/], ["s", DQ], ["s", SQ], ["n", NUM]] },
  make: { rules: [["c", /#.*$/], ["h", /^[\w./%-]+(?=\s*:(?!=))/], ["v", /\$\([^)]*\)|\$\{[^}]*\}|\$[@<^*?]/], ["s", DQ], ["s", SQ]] },
  css: { rules: [["c", /\/\*.*?\*\//], ["p", /[\w-]+(?=\s*:)/], ["s", DQ], ["s", SQ], ["n", /#[\da-fA-F]{3,8}\b|\b\d+(?:\.\d+)?(?:px|em|rem|%|s|ms|vh|vw)?\b/]], multi: [["c", /\/\*/, /\*\//]] },
  html: { rules: [["c", /<!--.*?-->/], ["k", /<\/?[\w-]+|\/?>/], ["p", /\b[\w-:]+(?==)/], ["s", DQ], ["s", SQ]], multi: [["c", /<!--/, /-->/]] },
  md: { rules: [["h", /^\s{0,3}#{1,6}\s.*$/], ["m", /^\s*(?:[-*+]|\d+[.)])\s/], ["s", /`[^`]+`/], ["k", /\*\*[^*]+\*\*/], ["v", /\[[^\]]+\]\([^)]+\)/]], multi: [["s", /^\s*```.*$/, /^\s*```\s*$/]] },
  diff: { line: (l) => (/^(?:\+\+\+|---)\s/.test(l) ? "h" : /^@@/.test(l) ? "m" : /^\+/.test(l) ? "a" : /^-/.test(l) ? "d" : /^(?:diff|index|new file|deleted file)\b/.test(l) ? "h" : "") },
};
const ALIAS = { bash: "sh", zsh: "sh", ts: "js", tsx: "js", jsx: "js", mjs: "js", cjs: "js", h: "c", cc: "c", cpp: "c", hpp: "c", cxx: "c", cs: "java",
  kt: "java", kts: "java", scala: "java", swift: "java", php: "c", yml: "yaml", jsonl: "json", ndjson: "json", patch: "diff", cfg: "ini", conf: "ini",
  env: "ini", properties: "ini", htm: "html", xml: "html", svg: "html", vue: "html", scss: "css", less: "css", markdown: "md", rst: "md", txt: null };

export const LANG_NAMES = { sh: "Shell", py: "Python", js: "JavaScript", go: "Go", rs: "Rust", c: "C/C++", java: "Java", rb: "Ruby", sql: "SQL", toml: "TOML",
  yaml: "YAML", json: "JSON", docker: "Dockerfile", ini: "Config", make: "Makefile", css: "CSS", html: "HTML/XML", md: "Markdown", diff: "Diff" };

// The language of a path, from its name or extension (and a shebang, when the first line is given).
export function langOf(path, firstLine = "") {
  const name = path.split("/").pop().toLowerCase();
  if (name === "dockerfile" || name.startsWith("dockerfile.") || name.endsWith(".dockerfile") || name === "containerfile") return "docker";
  if (name === "makefile" || name === "gnumakefile" || name.endsWith(".mk")) return "make";
  if (name === ".env" || name.startsWith(".env.")) return "ini";
  const ext = name.includes(".") ? name.split(".").pop() : "";
  if (ext in LANGS) return ext;
  if (ext in ALIAS) return ALIAS[ext];
  const bang = firstLine.match(/^#!.*\b(bash|sh|zsh|python3?|node|ruby)\b/);
  if (bang) return { bash: "sh", sh: "sh", zsh: "sh", python: "py", python3: "py", node: "js", ruby: "rb" }[bang[1]];
  return null;
}

function compile(lang) {
  if (lang._re !== undefined) return lang._re;
  const parts = [...(lang.multi || []).map((m) => m[1]), ...lang.rules.map((r) => r[1])];
  lang._classes = [...(lang.multi || []).map((m) => ["multi", m]), ...lang.rules.map((r) => ["rule", r[0]])];
  // one alternation, each rule in its own group, so the group that matched names the rule
  lang._re = new RegExp(parts.map((p) => `(${nocap(p.source)})`).join("|"), "g");
  return lang._re;
}
// a pattern's own groups made non-capturing, so group n of the alternation is always rule n
function nocap(src) {
  let out = "", inClass = false;
  for (let i = 0; i < src.length; i++) {
    const c = src[i];
    if (c === "\\") { out += c + (src[i + 1] ?? ""); i++; continue; }
    if (inClass) { if (c === "]") inClass = false; out += c; continue; }
    if (c === "[") { inClass = true; out += c; continue; }
    out += c === "(" && src[i + 1] !== "?" ? "(?:" : c;
  }
  return out;
}

const span = (cls, text) => (cls ? `<span class="tk-${cls}">${esc(text)}</span>` : esc(text));

// Highlight a whole file. Returns one HTML string per line.
export function highlight(text, lang) {
  const lines = text.split("\n");
  const L = LANGS[lang];
  if (!L) return lines.map(esc);
  if (L.line) return lines.map((l) => span(L.line(l), l));
  const re = compile(L);
  const out = [];
  let open = null;   // a token that runs on from an earlier line: {cls, end: RegExp}
  for (const line of lines) {
    let html = "", pos = 0;
    if (open) {
      const m = open.end.exec(line);
      if (!m) { out.push(span(open.cls, line)); continue; }
      const stop = m.index + m[0].length;
      html += span(open.cls, line.slice(0, stop));
      pos = stop;
      open = null;
    }
    re.lastIndex = pos;
    let m;
    while (pos < line.length && (m = re.exec(line))) {
      if (m[0] === "") { re.lastIndex++; continue; }
      const g = m.slice(1).findIndex((x) => x !== undefined);
      const [type, info] = L._classes[g];
      html += esc(line.slice(pos, m.index));
      if (type === "rule") { html += span(info, m[0]); pos = m.index + m[0].length; re.lastIndex = pos; continue; }
      // a multi-line opener: find its end on this line, or carry it to the next
      const [cls, , end] = info;
      if (end === "heredoc") {
        const word = m[0].match(/<<-?\s*['"]?(\w+)/)[1];
        html += span(cls, m[0]);
        open = { cls, end: new RegExp(`^\\s*${word}\\s*$`) };
        pos = line.length;
        break;
      }
      const from = m.index + m[0].length;
      const tail = line.slice(from);
      const e = end.exec(tail);
      if (e) { const stop = from + e.index + e[0].length; html += span(cls, line.slice(m.index, stop)); pos = stop; re.lastIndex = pos; continue; }
      html += span(cls, line.slice(m.index));
      open = { cls, end };
      pos = line.length;
      break;
    }
    html += esc(line.slice(pos));
    out.push(html);
  }
  return out;
}
