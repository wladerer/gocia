from __future__ import annotations

import numpy as np
import pytest


class TestIndividualModel:

    def test_default_weight_is_one(self, one_individual):
        assert one_individual.weight == pytest.approx(1.0)

    def test_mark_desorbed_zeros_weight(self, one_individual):
        desorbed = one_individual.mark_desorbed()
        assert desorbed.weight == pytest.approx(0.0)
        assert desorbed.desorption_flag is True
        assert desorbed.status == "desorbed"

    def test_mark_duplicate_zeros_weight(self, one_individual):
        dup = one_individual.mark_duplicate()
        assert dup.weight == pytest.approx(0.0)
        assert dup.status == "duplicate"

    def test_mark_isomer_sets_low_weight(self, one_individual):
        isomer = one_individual.mark_isomer(of_id="parent_id", isomer_weight=0.01)
        assert isomer.weight == pytest.approx(0.01)
        assert isomer.is_isomer is True
        assert isomer.isomer_of == "parent_id"
        assert isomer.status == "isomer"

    def test_mark_failed_zeros_weight(self, one_individual):
        failed = one_individual.mark_failed()
        assert failed.weight == pytest.approx(0.0)
        assert failed.status == "failed"

    def test_immutability_of_update_helpers(self, one_individual):
        """with_status should return a new object, not modify in place."""
        original_status = one_individual.status
        updated = one_individual.with_status("submitted")
        assert one_individual.status == original_status
        assert updated.status == "submitted"

    def test_is_selectable_for_converged(self, small_population):
        from gocia.population.individual import STATUS
        converged = [i for i in small_population if i.status == STATUS.CONVERGED]
        assert all(ind.is_selectable for ind in converged)

    def test_is_selectable_for_isomer(self, small_population):
        from gocia.population.individual import STATUS
        isomers = [i for i in small_population if i.status == STATUS.ISOMER]
        assert all(ind.is_selectable for ind in isomers)

    def test_not_selectable_for_desorbed_failed_pending(self, small_population):
        from gocia.population.individual import STATUS
        non_selectable_statuses = {STATUS.DESORBED, STATUS.FAILED, STATUS.PENDING}
        non_selectable = [i for i in small_population if i.status in non_selectable_statuses]
        assert all(not ind.is_selectable for ind in non_selectable)

    def test_from_parents_sets_genealogy(self, small_population):
        from gocia.population.individual import Individual, OPERATOR
        parent1, parent2 = small_population[0], small_population[1]
        child = Individual.from_parents(
            generation=1,
            parents=[parent1, parent2],
            operator=OPERATOR.SPLICE,
        )
        assert child.parent_ids == [parent1.id, parent2.id]
        assert child.operator == OPERATOR.SPLICE
        assert child.generation == 1

    def test_from_init_sets_init_operator(self):
        from gocia.population.individual import Individual, OPERATOR
        ind = Individual.from_init(generation=0)
        assert ind.operator == OPERATOR.INIT
        assert ind.parent_ids == []

    def test_status_constants_are_strings(self):
        from gocia.population.individual import STATUS
        assert isinstance(STATUS.PENDING, str)
        assert isinstance(STATUS.running_stage(1), str)
        assert isinstance(STATUS.converged_stage(2), str)

    def test_stage_number_extracted_correctly(self):
        from gocia.population.individual import STATUS
        assert STATUS.stage_number("running_stage_3") == 3
        assert STATUS.stage_number("converged_stage_1") == 1
        assert STATUS.stage_number("converged") is None

    def test_is_terminal_statuses(self):
        from gocia.population.individual import STATUS
        for s in (STATUS.CONVERGED, STATUS.DESORBED, STATUS.FAILED,
                  STATUS.DUPLICATE, STATUS.ISOMER):
            assert STATUS.is_terminal(s)
        assert not STATUS.is_terminal(STATUS.PENDING)
        assert not STATUS.is_terminal(STATUS.running_stage(1))

