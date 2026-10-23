import unittest

from windows_timer import TimerResolutionRequest


class _FakeCall:
    def __init__(self, result=0):
        self.result = result
        self.calls = []
        self.argtypes = None
        self.restype = None

    def __call__(self, milliseconds):
        self.calls.append(milliseconds)
        return self.result


class _FakeWinMM:
    def __init__(self, begin_result=0, end_result=0):
        self.timeBeginPeriod = _FakeCall(begin_result)
        self.timeEndPeriod = _FakeCall(end_result)


class WindowsTimerResolutionTests(unittest.TestCase):
    def test_off_by_default(self):
        self.assertIsNone(TimerResolutionRequest.from_environment({}))

    def test_environment_value_must_be_supported_integer(self):
        for value in ("0", "16", "one", "1.5"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                TimerResolutionRequest.from_environment(
                    {"EL_MODEM_WINDOWS_TIMER_RESOLUTION_MS": value}
                )

    def test_non_windows_request_is_reported_without_loading_winmm(self):
        def fail_if_loaded(_name):
            self.fail("WinMM must not be loaded on non-Windows")

        request = TimerResolutionRequest(1, platform="posix", library_loader=fail_if_loaded)
        request.start()
        self.assertEqual(request.report()["status"], "unsupported_platform")
        request.close()

    def test_windows_request_is_paired_once_and_reported(self):
        winmm = _FakeWinMM()
        request = TimerResolutionRequest(1, platform="nt", library_loader=lambda _name: winmm)
        request.start()
        self.assertEqual(request.report()["status"], "active")
        request.close()
        request.close()
        self.assertEqual(winmm.timeBeginPeriod.calls, [1])
        self.assertEqual(winmm.timeEndPeriod.calls, [1])
        self.assertEqual(request.report()["status"], "released")
        self.assertEqual(request.report()["begin_result"], 0)
        self.assertEqual(request.report()["end_result"], 0)

    def test_failed_request_does_not_claim_active_timer(self):
        winmm = _FakeWinMM(begin_result=97)
        request = TimerResolutionRequest(1, platform="nt", library_loader=lambda _name: winmm)
        with self.assertRaises(OSError):
            request.start()
        self.assertEqual(request.report()["status"], "request_failed")
        self.assertEqual(winmm.timeEndPeriod.calls, [])


if __name__ == "__main__":
    unittest.main()
