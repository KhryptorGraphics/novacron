/**
 * Package entry point: the fabric surface must be importable from the SDK root
 * (@novacron/dwcp-sdk), not only from its module path.
 */

import * as sdk from '../src/index';
import { FabricAPIError, FabricClient } from '../src/fabric';

describe('package exports', () => {
  it('exposes the fabric client, error, and default timeout', () => {
    expect(sdk.FabricClient).toBe(FabricClient);
    expect(sdk.FabricAPIError).toBe(FabricAPIError);
    expect(sdk.DEFAULT_FABRIC_TIMEOUT).toBe(60000);
  });
});