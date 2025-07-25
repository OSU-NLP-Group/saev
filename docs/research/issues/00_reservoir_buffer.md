# Reservoir Buffer Failing Tests

uv run pytest tests/test_reservoir_buffer.py -xvs -k 'test_across_process and thread'

Specifically, `test_across_process_visibility` with the 'thread' backend.

tests/test_reservoir_buffer.py
src/saev/data/buffers.py

This test times out.
It used to pass.
I think the issue is that I changed from free.acquire() on at a time to batching it:

```diff
-        for x, m in zip(xs, metas_it):
+        for _ in range(n):
             self.free.acquire()  # block if full
-            with self.lock:
-                idx = self.size.value  # append at tail
-                self.data[idx].copy_(x)
-                self.meta[idx] = m
-                self.size.value += 1
+
+        with self.lock:
+            start = self.size.value
+            end = start + n
+            self.data[start:end].copy_(xs)
+            self.meta[start:end].copy_(metadata)
+            self.size.value = end
+
+        for _ in range(n):
             self.full.release()
```
