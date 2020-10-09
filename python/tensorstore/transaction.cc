#include "python/tensorstore/transaction.h"

#include "python/tensorstore/future.h"
#include "pybind11/pybind11.h"
#include "tensorstore/transaction.h"

namespace tensorstore {
namespace internal_python {

namespace py = ::pybind11;

void RegisterTransactionBindings(pybind11::module m) {
  using internal::TransactionState;
  py::class_<TransactionState, TransactionState::CommitPtr> cls_transaction(
      m, "Transaction", R"(

Transactions are used to stage a group of modifications (e.g. writes to
`TensorStore` objects) in memory, and then either commit the group all at once
or abort it.

Two transaction modes are currently supported:

"Isolated" transactions provide write isolation: no modifications made are
visible or persist outside the transactions until the transaction is committed.
In addition to allowing modifications to be aborted/rolled back, this can also
improve efficiency by ensuring multiple writes to the same underlying storage
key are coalesced.

"Atomic isolated" transactions have all the properties of "isolated"
transactions but additionally guarantee that all of the modifications will be
committed atomically, i.e. at no point will an external reader observe only some
but not all of the modifications.  If the modifications made in the transaction
cannot be committed atomically, the transaction will fail (without any changes
being made).

Example usage:

    >>> txn = ts.Transaction()
    >>> store = ts.open({
    ...     'driver': 'n5',
    ...     'kvstore': {
    ...         'driver': 'file',
    ...         'path': '/tmp/dataset'
    ...     },
    ...     'metadata': {
    ...         'dataType': 'uint16',
    ...         'blockSize': [2, 3],
    ...         'dimensions': [5, 6],
    ...         'compression': {
    ...             'type': 'raw'
    ...         }
    ...     },
    ...     'create': True,
    ...     'delete_existing': True
    ... }).result()
    >>> store.with_transaction(txn)[1:4, 2:5] = 42
    >>> store.with_transaction(txn)[0:2, 4] = 43

Uncommitted changes made in a transaction are visible from a transactional read
using the same transaction, but not from a non-transactional read:

    >>> store.with_transaction(txn).read().result()
    array([[ 0,  0,  0,  0, 43,  0],
           [ 0,  0, 42, 42, 43,  0],
           [ 0,  0, 42, 42, 42,  0],
           [ 0,  0, 42, 42, 42,  0],
           [ 0,  0,  0,  0,  0,  0]], dtype=uint16)
    >>> store.read().result()
    array([[0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0]], dtype=uint16)

The transaction can be committed using
:py:obj:`tensorstore.Transaction.commit_async`.

    >>> txn.commit_async().result()
    >>> store.read().result()
    array([[ 0,  0,  0,  0, 43,  0],
           [ 0,  0, 42, 42, 43,  0],
           [ 0,  0, 42, 42, 42,  0],
           [ 0,  0, 42, 42, 42,  0],
           [ 0,  0,  0,  0,  0,  0]], dtype=uint16)

The :py:obj:`tensorstore.Transaction` class can also be used as a regular or
asynchronous context manager:

    >>> with ts.Transaction() as txn:
    ...     store.with_transaction(txn)[0:2, 1:3] = 44
    ...     store.with_transaction(txn)[0, 0] = 45
    >>> store.read().result()
    array([[45, 44, 44,  0, 43,  0],
           [ 0, 44, 44, 42, 43,  0],
           [ 0,  0, 42, 42, 42,  0],
           [ 0,  0, 42, 42, 42,  0],
           [ 0,  0,  0,  0,  0,  0]], dtype=uint16)

    >>> async with ts.Transaction() as txn:
    ...     store.with_transaction(txn)[0:2, 1:3] = 44
    ...     store.with_transaction(txn)[0, 0] = 45
    >>> await store.read()
    array([[45, 44, 44,  0, 43,  0],
           [ 0, 44, 44, 42, 43,  0],
           [ 0,  0, 42, 42, 42,  0],
           [ 0,  0, 42, 42, 42,  0],
           [ 0,  0,  0,  0,  0,  0]], dtype=uint16)

If the block exits normally, the transaction is committed automatically.  If the
block raises an exception, the transaction is aborted.

)");

  cls_transaction.def(
      py::init([](bool atomic) {
        return TransactionState::ToCommitPtr(Transaction(
            atomic ? tensorstore::atomic_isolated : tensorstore::isolated));
      }),
      py::arg("atomic") = false, R"(Creates a new transaction.)");
  cls_transaction.def(
      "commit_async",
      [](const TransactionState::CommitPtr& self) {
        self->RequestCommit();
        return self->future();
      },
      R"(Asynchronously commits the transaction.

Has no effect if the `tensorstore.Transaction.commit_async` or
`tensorstore.Transaction.abort` has already been called.

Returns the associated `tensorstore.Transaction.future`, which may be used to
check if the commit was successful.
)");
  cls_transaction.def(
      "commit_sync",
      [](const TransactionState::CommitPtr& self) {
        self->RequestCommit();
        return self->future();
      },
      R"(Synchronously commits the transaction.

Equivalent to:

    self.commit_async().result()

Returns `None` if the commit is successful, and raises an error otherwise.
)");
  cls_transaction.def(
      "abort",
      [](const TransactionState::CommitPtr& self) { self->RequestAbort(); },
      R"(Aborts the transaction.

Has no effect if the `tensorstore.Transaction.commit_async` or
`tensorstore.Transaction.abort` has already been called.
)");
  cls_transaction.def_property_readonly(
      "future",
      [](const TransactionState::CommitPtr& self) { return self->future(); },
      R"(Commit result future.

Becomes ready when the transaction has either been committed successfully or
aborted.
)");
  cls_transaction.def_property_readonly(
      "aborted",
      [](const TransactionState::CommitPtr& self) { return self->aborted(); },
      "Indicates whether the transaction has been aborted.");
  cls_transaction.def_property_readonly(
      "commit_started",
      [](const TransactionState::CommitPtr& self) {
        return self->commit_started();
      },
      "Indicates whether the commit of the transaction has already started.");

  cls_transaction.def_property_readonly(
      "atomic",
      [](const TransactionState::CommitPtr& self) {
        return self->mode() == tensorstore::atomic_isolated;
      },
      "Indicates whether the transaction is atomic.");

  cls_transaction.def_property_readonly(
      "open",
      [](const TransactionState::CommitPtr& self) {
        return !self->commit_started() && !self->aborted();
      },
      R"(Indicates whether the transaction is still open.

The transaction remains open until commit starts or it is aborted.  Once commit
starts or it has been aborted, it may not be used for any additional
transactional operations.
)");

  cls_transaction.def("__enter__", [](const TransactionState::CommitPtr& self) {
    return self;
  });
  cls_transaction.def("__exit__", [](const TransactionState::CommitPtr& self,
                                     py::object exc_type, py::object exc_value,
                                     py::object traceback) {
    if (exc_value.ptr() == Py_None) {
      // Block exited normally.  Commit the transaction.
      self->RequestCommit();
      return ValueOrThrow(internal_python::InterruptibleWait(self->future()));
    } else {
      // Block exited with an exception.  Abort the transaction.
      self->RequestAbort();
      internal_python::InterruptibleWait(self->future());
    }
  });
  cls_transaction.def(
      "__aenter__",
      [](const TransactionState::CommitPtr& self)
          -> Future<const TransactionState::CommitPtr> {
        // Hack to obtain an "awaitable" that returns `self`.
        return MakeReadyFuture<TransactionState::CommitPtr>(self);
      });
  cls_transaction.def(
      "__aexit__",
      [](const TransactionState::CommitPtr& self, py::object exc_type,
         py::object exc_value, py::object traceback) -> Future<const void> {
        if (exc_value.ptr() == Py_None) {
          // Block exited normally.  Commit the transaction.
          self->RequestCommit();
          return self->future();
        } else {
          // Block exited with an exception.  Abort the transaction.
          self->RequestAbort();
          // Wait for `self->future()` to become ready, but ignore the result.
          return MapFuture(
              InlineExecutor{},
              [](const Result<void>& result) { return MakeResult(); },
              self->future());
        }
      });
}

}  // namespace internal_python
}  // namespace tensorstore
