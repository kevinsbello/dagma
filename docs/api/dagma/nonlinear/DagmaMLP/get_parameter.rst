:py:meth:`dagma.nonlinear.DagmaMLP.get_parameter <dagma.nonlinear.DagmaMLP.get_parameter>`
==========================================================================================
.. _dagma.nonlinear.DagmaMLP.get_parameter:
.. py:method:: get_parameter(target: str) -> torch.nn.parameter.Parameter

   Returns the parameter given by ``target`` if it exists,
   otherwise throws an error.

   See the docstring for ``get_submodule`` for a more detailed
   explanation of this method's functionality as well as how to
   correctly specify ``target``.

   :param target: The fully-qualified string name of the Parameter
                  to look for. (See ``get_submodule`` for how to specify a
                  fully-qualified string.)

   :returns: The Parameter referenced by ``target``
   :rtype: torch.nn.Parameter

   :raises AttributeError: If the target string references an invalid
       path or resolves to something that is not an
       ``nn.Parameter``

