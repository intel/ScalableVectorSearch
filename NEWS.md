# SVS 0.5.1 Release Notes

## Additions and Changes

* Fixed reduced recall of filtered search at low filter pass rates. When the batch iterator resumes, it now resets the visited set to the current search buffer contents, so candidates dropped from the buffer can be found again (#437)
