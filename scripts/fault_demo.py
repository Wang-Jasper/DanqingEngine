# ============================================================================
# fault_demo.py - Phase 13.8 verification script: intentional exceptions.
# ----------------------------------------------------------------------------
# Purpose:
#   Verify that ScriptEngine isolates a Python exception to the offending
#   Entity, mirrors the traceback into the Script Console with a per-entity
#   prefix, and keeps the engine + every other ScriptComponent running.
#
# How to use:
#   1) In the Inspector, attach a ScriptComponent to any entity (e.g. a
#      second Cube). Click "Load Script (.py)" and pick this file.
#   2) Click Play. Within a few frames the script raises:
#        - on_start: prints to stdout (verifies redirect + prefix)
#        - on_update first call: also prints
#        - on_update second call onwards: raises RuntimeError
#   3) Open the Script Console. Expected output:
#        [stdout] [on_start #N <Name>] fault_demo: on_start fired
#        [stdout] [on_update #N <Name>] fault_demo: about to raise!
#        [ScriptEngine] Python error in on_update[entity #N]:
#        Traceback (most recent call last):
#          ...
#        RuntimeError: intentional Phase 13.8 fault
#      The originating entity stops calling on_update; OTHER scripts (e.g.
#      spin_cube.py on the default Cube) keep running unaffected.
#   4) Click the Inspector "Reload" button on the faulted entity to clear
#      the fault and re-attach (faulted state survives only until reload).
# ============================================================================

# Note: we deliberately do NOT `import engine` or anything else at module
# load time, because we want the file to be safely top-level loadable. All
# error injection happens inside on_*() so attaching the script never crashes
# the engine — the user has to actively press Play to see the failure.

# Module-level call counter so we raise on the *second* on_update call.
# That gives us at least one "good" frame in the console before the fault,
# which makes the prefix-vs-traceback layout obvious in the Script Console.
_update_count = 0


def on_start(self):
    # Plain print() goes through sys.stdout, which Phase 13.8 redirected to
    # ScriptEngine::writeStdoutChunk. Output should appear in the Script
    # Console with prefix "[stdout] [on_start #<id> <Name>]".
    print("fault_demo: on_start fired (stdout redirect + entity prefix verified)")


def on_update(self, dt):
    global _update_count
    _update_count += 1

    if _update_count == 1:
        # First update tick: emit normal output to confirm the redirect path
        # works while a callback is running. Also exercise stderr.
        print("fault_demo: about to raise!")
        # sys.stderr is also redirected; this proves stderr gets the
        # "[stderr]" prefix in the console.
        import sys
        print("fault_demo: this line should appear with [stderr] prefix",
              file=sys.stderr)
        return

    # Second tick onwards: raise. ScriptEngine catches the exception inside
    # callOnUpdateAll, sets es.faulted=true (and ScriptComponent.faulted=true
    # for the Inspector badge), captures the traceback into logBuffer, then
    # silently skips this entity's callbacks until the user clicks Reload.
    raise RuntimeError("intentional Phase 13.8 fault")


def on_stop(self):
    # If the user pressed Stop after the fault was raised, this won't be
    # called for THIS entity (es.faulted skips on_stop). Stop on a non-faulted
    # entity still hits this line — useful for verifying the "fault isolation"
    # claim across multiple ScriptComponents.
    print("fault_demo: on_stop reached (only fires if no exception happened)")
