# Automation Access Catalog Parity

Date: 2026-07-17

Status: implemented

## Problem

OAuth consent for an MCP resource showed the configured named-service
namespace and operation tree. The manual **Create automation access** surface
showed only the resource's grant chips. A user therefore could not see which
named-service operations those grants enabled or which connected account the
provider would require later.

## Implementation

`AutomationAccessService.resource_options()` now returns, for each resource
with `named_services` configuration:

```text
resources[].named_services[]
  namespace / label / description / authority_id
  tools
    operation / label / description / grants
    operations
      <provider-declared operation> / label / description / grants
  connected_accounts
    <provider discovery metadata copied without normalization>
```

The namespace/tool tree is produced by `NamedServiceBoundaryCatalog`, the same
descriptor-backed catalog used by OAuth consent. `connected_accounts` comes
from the live named-service provider's discovery metadata. Flat provider claim
sets stay flat; a provider's `claims_by_operation` mapping stays nested. The
projection does not synthesize action variants.

The Connections widget renders this tree under the MCP resource. Every
namespace operation is selectable. Selecting an operation also selects the
platform grants declared by that operation; removing a required grant removes
the affected operation selection.

The create request carries the exact existing operation identifiers as:

```text
named_service_operations
  <resource>
    <namespace>
      [<operation>, ...]
```

`create_access()` validates each identifier and its grants against the live
resource descriptor, then narrows the existing `named_services` policy tree to
the selected subset before binding the token in `GrantStore`. Provider
requirements reuse the existing **Delegated to KDCube** account catalog and
consent deep link.

## Invariants

- No new registry, grant entity, action id, or token format was introduced.
- The automation bearer contains KDCube delegation only; it never contains a
  Gmail, Slack, or other provider credential.
- `create_access()` stores a narrowed copy of the configured `named_services`
  map in `GrantStore`. Discovery-only presentation metadata is not persisted
  into the credential policy.
- KDCube Services already prefers that stored policy when constructing its
  `NamedServiceBoundaryCatalog`; unselected namespaces and operations are
  therefore denied by the normal runtime bridge.
- Demand-driven connected-account consent is unchanged. The automation bearer
  contains no provider token.
- If provider discovery is unavailable, the configured namespace operation
  catalog still renders; only connected-account readiness is omitted.

## Verification

- Backend regression verifies exact namespace projection, exact provider
  requirement pass-through, duplicate removal, and absence of synthesized
  `object.action.post_message` operations.
- The regression also verifies discovery presentation metadata does not enter
  the persisted grant policy.
- Backend regressions verify that a search-only selection omits action tools
  and that an operation cannot be selected without its declared grants.
- The Connections widget passes TypeScript checking and a production Vite
  build.

## Later clarification

The 2026-07-27 descriptor update declares exact action variants such as
`object.action.send` and `object.action.update_values`. Connection Hub projects
those declared identifiers unchanged; it still does not synthesize variants.
See
[2026-07-27 - Exact named-service actions and Sheets](2026-07-27-exact-named-service-actions-and-sheets.md).
