# TSX Generator Constraints

## Critical Rules

**Output:** TypeScript React (TSX) code that compiles with Babel Standalone to standalone HTML with CDN libraries.

## ❌ Forbidden

**Libraries:**
- Recharts (use Chart.js instead)
- Material-UI, Ant Design, any npm imports
- No `import` statements (React loaded via CDN)
- No `const { useState, ... } = React;` (already provided)
- No `export default` (not needed)

**TypeScript Operators (Babel Standalone doesn't support):**
```typescript
// ❌ Non-null assertion
document.getElementById('root')!

// ❌ Const assertions  
const x = { y: 1 } as const;

// ❌ Enums
enum Status { Active }
```

## ✅ Use Instead

```typescript
// ✅ Null checks
const el = document.getElementById('root');
if (el) { const root = ReactDOM.createRoot(el); }

// ✅ Regular const
const x = { y: 1 };

// ✅ Union types
type Status = 'active' | 'inactive';
```

## Required Patterns

**Charts:** Use Chart.js with canvas + useRef pattern, always destroy in cleanup

**Styling:** Tailwind utility classes only

**React:** `const { useState, useEffect, useRef, useMemo, useCallback } = React;`

**Render:** Always use null check before createRoot

That's it.