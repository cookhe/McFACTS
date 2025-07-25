===============
Commit Messages
===============

See the `Contributing Guide <https://github.com/McFACTS/McFACTS/blob/main/docs/source/contribute.rst>`_ for more
information on making changes to McFACTS.

`Conventional Commits <https://www.conventionalcommits.org/en/v1.0.0/>`_ is a formatting convention that enables
us to easily

#. review the code's history
#. automatically generate our changelog

Please follow these guidelines when committing changes as you work on McFACTS.

Formatting
**********
.. code-block::

   <type>[optional scope]: <description>

   [optional body]

   [optional footer(s)]

Message Subject
^^^^^^^^^^^^^^^

The first line cannot be longer than 70 characters. The second line is always blank, and other lines should be no
longer than 80 characters.
The ``type`` and ``scope`` must be lower case.

**<type> must be one of:**

* ``feat`` – a new feature is introduced with the changes
* ``fix`` – a bug fix has occurred
* ``chore`` – changes that do not relate to a fix or feature and don't modify src or test files (for example updating dependencies)
* ``refactor`` – refactored code that neither fixes a bug nor adds a feature
* ``docs`` – updates to documentation such as a the README or other markdown files
* ``style`` – changes that do not affect the meaning of the code, likely related to code formatting such as white-space, missing semi-colons, and so on.
* ``test`` – including new or correcting previous tests
* ``perf`` – performance improvements
* ``ci`` – continuous integration related
* ``build`` – changes that affect the build system or external dependencies
* ``revert`` – reverts a previous commit

**[scope] examples:**

* ``phy`` - physics
* ``io`` - input/output
* ``vis`` - visualization
* ``setup``
* ``ext`` - external module interfaces
* etc.

**<description> should be:**

#. present tense
#. gramatically structured to complete the sentence, "Making this change will _____."


The ``[optional body]`` can be used to include additional details when necessary, and \
``[optional footer(s)]`` should be used when the change addresses a reported issue, ``Closes #123``).

See References and Resources for examples.

References and Resources:
*************************

#. `https://www.conventionalcommits.org <https://www.conventionalcommits.org/en/v1.0.0/>`_
#. `Semantic Commit Messages <https://gist.github.com/joshbuchea/6f47e86d2510bce28f8e7f42ae84c716>`_
#. `<https://sparkbox.com/foundry/semantic_commit_messages>`_
#. `<https://karma-runner.github.io/1.0/dev/git-commit-msg.html>`_

