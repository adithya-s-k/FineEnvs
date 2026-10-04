import test from 'node:test';
import assert from 'node:assert/strict';
import { schemaForm, readForm, resolveSchema, validateValue } from '../web/js/schemaform.js';

const form = (fields = [], root = null) => ({
  querySelector: () => root,
  querySelectorAll: () => fields.map(([k, t, value]) => ({ dataset: { k, t }, value })),
});
const plain = (v) => JSON.parse(JSON.stringify(v));

test('optional booleans can be omitted, and explicit false remains false', () => {
  const schema = { type: 'object', properties: { enabled: { type: 'boolean', default: true } } };
  assert.deepEqual(plain(readForm(form([['enabled', 'bool', '']]), schema)), {});
  assert.deepEqual(plain(readForm(form([['enabled', 'bool', 'false']]), schema)), { enabled: false });
  assert.match(schemaForm(schema), /Use server default/);
});

test('integer inputs reject fractions rather than truncating actions', () => {
  const s = { type: 'object', properties: { action: { type: 'integer', minimum: 0, maximum: 3 } }, required: ['action'] };
  for (const value of ['1.5', '4', '-1', 'Infinity', '9007199254740992', '']) {
    assert.throws(() => readForm(form([['action', 'int', value]]), s));
  }
  assert.equal(readForm(form([['action', 'int', '0']]), s).action, 0);
});

test('enum values retain their JSON type and never silently pick the first option', () => {
  const s = { type: 'object', properties: { action: { enum: [0, 1, 'stop', null] } }, required: ['action'] };
  assert.match(schemaForm(s), /Choose a value/);
  assert.equal(readForm(form([['action', 'enum', '0']]), s).action, 0);
  assert.equal(readForm(form([['action', 'enum', 'null']]), s).action, null);
  assert.throws(() => readForm(form([['action', 'enum', '"bad"']]), s));
});

test('local references retain defaults and referenced discriminators', () => {
  const s = { type: 'object', $defs: { Mode: { const: 'move' }, Count: { type: 'integer', default: 7 } },
    properties: { type: { $ref: '#/$defs/Mode' }, count: { $ref: '#/$defs/Count' } } };
  assert.match(schemaForm(s), /value="7"/);
  assert.equal(readForm(form(), s).type, 'move');
});

test('recursive and missing references are bounded', () => {
  const s = { $ref: '#/$defs/Loop', $defs: { Loop: { $ref: '#/$defs/Loop' } } };
  assert.doesNotThrow(() => schemaForm(s));
  assert.deepEqual(resolveSchema({ $ref: '#/$defs/missing' }, s), {});
});

test('root action unions use JSON without losing nested arguments', () => {
  const s = { oneOf: [
    { type: 'object', properties: { type: { const: 'move' }, position: { type: 'array', items: { type: 'number' }, minItems: 2 } }, required: ['type', 'position'] },
    { type: 'object', properties: { type: { const: 'stop' } }, required: ['type'] },
  ] };
  assert.match(schemaForm(s), /data-root-json/);
  assert.deepEqual(readForm(form([], { value: '{"type":"move","position":[1,2]}' }), s), { type: 'move', position: [1, 2] });
  assert.throws(() => readForm(form([], { value: '{"type":"move"}' }), s));
});

test('open objects have a JSON editor rather than claiming no inputs', () => {
  assert.match(schemaForm({ type: 'object' }), /data-root-json/);
  assert.match(schemaForm({ type: 'object', additionalProperties: false }), /No inputs/);
});

test('all interpolated upstream schema values are escaped', () => {
  const payload = '\"><img src=https://evil.test/>';
  const s = { type: 'object', properties: { [payload]: { type: 'number', default: payload, minimum: payload } } };
  const html = schemaForm(s, { prefix: payload });
  assert.ok(!html.includes('<img'));
  assert.ok(!html.includes('min="' + payload));
});

test('sanitized property names cannot create duplicate input ids', () => {
  const html = schemaForm({ properties: { 'a.b': { type: 'string' }, 'a/b': { type: 'string' } } });
  const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map((x) => x[1]);
  assert.equal(ids.length, new Set(ids).size);
});

test('prototype-like field names remain ordinary JSON properties', () => {
  const schema = JSON.parse('{"type":"object","properties":{"__proto__":{"type":"object"}}}');
  const result = readForm(form([['__proto__', 'json', '{"polluted":true}']]), schema);
  assert.equal(Object.getPrototypeOf(result), null);
  assert.equal({}.polluted, undefined);
  assert.equal(JSON.parse(JSON.stringify(result)).__proto__.polluted, true);
});

test('nested required, size and additional-properties constraints report actionable errors', () => {
  const s = { type: 'object', properties: { args: { type: 'object', properties: { text: { type: 'string', minLength: 2 } }, required: ['text'], additionalProperties: false } } };
  assert.throws(() => validateValue({ args: {} }, s), /args.text is required/);
  assert.throws(() => validateValue({ args: { text: '' } }, s), /at least 2/);
  assert.throws(() => validateValue({ args: { text: 'ok', answer: 'secret' } }, s), /not an allowed field/);
});
