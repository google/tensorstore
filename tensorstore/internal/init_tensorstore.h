
#ifndef TENSORSTORE_INTERNAL_INIT_TENSORSTORE_H_
#define TENSORSTORE_INTERNAL_INIT_TENSORSTORE_H_

namespace tensorstore {

// InitTensorstore allows integration with google internal flag parsing, which
// differs slightly from absl flag parsing. Opensource users should not use
// this, and just rely on absl::ParseCommandLine directly.
void InitTensorstore(int* argc, char*** argv);

}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_INIT_TENSORSTORE_H_
