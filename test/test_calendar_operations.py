"""Unit tests for calendar operation dispatch and action execution."""

import unittest
from unittest.mock import patch

import calendar_reminder as calendar


class CalendarOperationTests(unittest.TestCase):
    def test_view_uses_operation_window_and_timezone(self):
        operation = calendar.CalendarOperation(
            action="view",
            date_from="2026-11-28",
            date_to="2026-11-30",
            time_zone="America/Toronto",
        )
        expected = [{"Chunk Name": "Calendar events"}]
        with (
            patch.object(calendar, "view_calendar_events", return_value=[{"id": "e1"}]) as view,
            patch.object(calendar, "calendar_events_to_chunks", return_value=expected) as to_chunks,
        ):
            result = calendar.execute_calendar_operation(operation)

        self.assertEqual(result, expected)
        view.assert_called_once_with(
            date_from="2026-11-28",
            date_to="2026-11-30",
            time_zone="America/Toronto",
        )
        self.assertIn("Reference date:", to_chunks.call_args.kwargs["result_note"])

    def test_create_missing_fields_returns_prompt_chunk(self):
        operation = calendar.CalendarOperation(action="create", summary="Meeting")
        expected = [{"Chunk Name": "Calendar events"}]
        with (
            patch.object(calendar, "calendar_events_to_chunks", return_value=expected) as to_chunks,
            patch.object(calendar, "add_calendar_event") as create,
        ):
            result = calendar.execute_calendar_operation(operation)

        self.assertEqual(result, expected)
        to_chunks.assert_called_once_with(
            [], result_note="Please provide an event title, date, start time, and end time."
        )
        create.assert_not_called()

    def test_create_uses_local_wall_clock_and_returns_update_chunk(self):
        operation = calendar.CalendarOperation(
            action="create",
            summary="Meeting",
            start_iso="2026-11-28T11:00:00",
            end_iso="2026-11-28T12:00:00",
            time_zone="America/Toronto",
        )
        expected = [{"Chunk Name": "Calendar Updated"}]
        with (
            patch.object(calendar, "add_calendar_event", return_value={"success": True}) as create,
            patch.object(calendar, "calendar_updated_to_chunk", return_value=expected[0]),
        ):
            result = calendar.execute_calendar_operation(operation)

        self.assertEqual(result, expected)
        create.assert_called_once_with(
            "Meeting",
            "2026-11-28T11:00:00",
            "2026-11-28T12:00:00",
            "America/Toronto",
            interpret_as_local_time=True,
        )

    def test_delete_uses_timezone_and_deletes_matching_event(self):
        operation = calendar.CalendarOperation(
            action="delete",
            summary="Meeting",
            date_from="2026-11-28",
            date_to="2026-11-28",
            time_zone="America/Toronto",
        )
        event = {"id": "e1", "summary": "Meeting"}
        expected = [{"Chunk Name": "Calendar events"}]
        with (
            patch.object(calendar, "view_calendar_events", return_value=[event]) as view,
            patch.object(calendar, "delete_calendar_event") as delete,
            patch.object(calendar, "calendar_events_to_chunks", return_value=expected) as to_chunks,
        ):
            result = calendar.execute_calendar_operation(operation)

        self.assertEqual(result, expected)
        view.assert_called_once_with(
            date_from="2026-11-28",
            date_to="2026-11-28",
            time_zone="America/Toronto",
        )
        delete.assert_called_once_with("e1")
        self.assertEqual(to_chunks.call_args.args[0], [event])

    def test_delete_without_events_returns_hint_without_deleting(self):
        operation = calendar.CalendarOperation(action="delete", time_zone="America/Toronto")
        expected = [{"Chunk Name": "Calendar events"}]
        with (
            patch.object(calendar, "view_calendar_events", return_value=[]),
            patch.object(calendar, "delete_calendar_event") as delete,
            patch.object(calendar, "calendar_events_to_chunks", return_value=expected) as to_chunks,
        ):
            result = calendar.execute_calendar_operation(operation)

        self.assertEqual(result, expected)
        delete.assert_not_called()
        to_chunks.assert_called_once_with(
            [], result_note="There are no upcoming events to delete."
        )


if __name__ == "__main__":
    unittest.main()
