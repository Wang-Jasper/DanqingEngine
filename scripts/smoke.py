# ============================================================================
# smoke.py - Phase 13.1 smoke test for the embedded Python interpreter.
# ----------------------------------------------------------------------------
# This script is auto-loaded by DeferredRenderer::init() at engine start-up.
# Its only job is to prove that:
#   1. The CPython interpreter is up.
#   2. The embedded `engine` module is importable.
#   3. C++ -> Python -> C++ round-trip works (engine.log writes into the
#      ScriptEngine log buffer, which is mirrored to stdout for now).
#
# Phase 13.2+ will replace this with a real binding test.
# ============================================================================

from engine import log

log("hello from python")
log("ScriptEngine smoke test OK")
