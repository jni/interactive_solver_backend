import collections
import json

identifier_to_class = collections.defaultdict(lambda : None, **{
    # Detach.identifier         : Detach,
    # Merge.identifier          : Merge,
    # MergeAndDetach.identifier : MergeAndDetach
    })

class Action(object):

    def __init__(self, identifier):

        super(Action, self).__init__()

        self.identifier = identifier

    def identifier(self):
        return self.identifier

    def data(self):
        return {}

    def to_json(self):
        return json.dumps({
            'type' : self.identifier,
            'data' : self.data()
            })

    def to_json_array(*actions):
        array = []
        for action in actions:
            array.append(json.loads(action.to_json()))

        return json.dumps(array)

    def from_json_array(data):

        if isinstance(data, str):
            data = json.loads(data)

        actions = []
        if not isinstance(data, collections.Sequence):
            return actions

        for d in data:
            if isinstance(d, dict) and 'type' in d.keys():
                class_type = identifier_to_class[d['type']]
                if class_type:
                    actions.append(class_type.from_data(d['data']))

        return actions

class Detach(Action):

    identifier = 'separate'

    def __init__(self, fragment_id, *detach_from):
        super(Detach, self).__init__(Detach.identifier)
        self.fragment_id = fragment_id
        self.detach_from = detach_from

    def data(self):
        return {
            'fragment' : self.fragment_id,
            'from'     : self.detach_from
            }

    def from_data(data):
        return Detach(data['fragment'], *data['from'])

class Merge(Action):

    identifier = 'merge'

    def __init__(self, *ids):
        super(Merge, self).__init__(Merge.identifier)
        self.ids = ids

    def data(self):
        return {
            'fragments' : self.ids
            }

    def from_data(data):
        return Merge(*data['fragments'])


class MergeAndDetach(Action):

    identifier = 'merge-and-separate'

    def __init__(self, merge_ids, detach_from):
        super(MergeAndDetach, self).__init__(MergeAndDetach.identifier)

        def ensure_sequence(parameter):
            return () if parameter is None else (parameter if isinstance(parameter, collections.Sequence) else (parameter,))

        self.merge_ids   = ensure_sequence(merge_ids)
        self.detach_from = ensure_sequence(detach_from)

    def data(self):
        return {
            "fragments" : self.merge_ids,
            "from"      : self.detach_from
            }

    def from_data(data):
        return MergeAndDetach(data['fragments'], data['from'])

class ConfirmGroupings(Action):

    identifier = 'confirm-multiple-segments'

    def __init__(self, *merge_ids):
        super(ConfirmGroupings, self).__init__(ConfirmGroupings.identifier)

        self.merge_ids = merge_ids

    def data(self):
        return {
            "fragments" : self.merge_ids,
            }

    def from_data(data):
        return ConfirmGroupings(*data['fragments'])


identifier_to_class[Detach.identifier]           = Detach
identifier_to_class[Merge.identifier]            = Merge
identifier_to_class[MergeAndDetach.identifier]   = MergeAndDetach
identifier_to_class[ConfirmGroupings.identifier] = ConfirmGroupings


if __name__ == '__main__':
    detach           = Detach(1, 2, 3, 4)
    merge            = Merge(5, 6, 7)
    merge_and_detach = MergeAndDetach((1, 2, 3), (4, 5, 6))
    confirm_grouping = ConfirmGroupings((1, 2, 3), (4, 5, 6))

    json_array = Action.to_json_array(detach, merge, merge_and_detach, confirm_grouping)
    data_array = Action.from_json_array(json_array)

    print(json_array)

    print(detach.to_json())
    print(data_array[0].to_json())
    print(merge.to_json())
    print(data_array[1].to_json())
    print(merge_and_detach.to_json())
    print(data_array[2].to_json())
    print(confirm_grouping.to_json())
    print(data_array[3].to_json())
