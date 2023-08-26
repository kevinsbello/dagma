.. py:method:: get_buffer(target: str) -> torch.Tensor

   Returns the buffer given by ``target`` if it exists,
   otherwise throws an error.

   See the docstring for ``get_submodule`` for a more detailed
   explanation of this method's functionality as well as how to
   correctly specify ``target``.

   :param target: The fully-qualified string name of the buffer
                  to look for. (See ``get_submodule`` for how to specify a
                  fully-qualified string.)

   :returns: The buffer referenced by ``target``
   :rtype: torch.Tensor

   :raises AttributeError: If the target string references an invalid
       path or resolves to something that is not a
       buffer

