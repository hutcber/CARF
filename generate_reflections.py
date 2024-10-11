from utils import get_completion
from utils_qwen import qwen_completion
from utils_router import router_completion
import os
import json

from typing import List, Dict, Any

with open("./reflexion_few_shot_examples.txt", 'r') as f:
    FEW_SHOT_EXAMPLES = f.read()

FOLDER = './prompts'
PROMPT_FILE = 'data.json'
with open(os.path.join(FOLDER, PROMPT_FILE), 'r') as f:
    demo = json.load(f)

PROMPT_FILE = 'alfworld_3prompts.json'
with open(os.path.join(FOLDER, PROMPT_FILE), 'r') as f:
    d = json.load(f)

def _get_scenario(s: str) -> str:
    """Parses the relevant scenario from the experience log."""
    return s.split("Here is the task:")[-1].strip()


def _generate_reflection_query(log_str: str, memory: List[str]) -> str:
    """Allows the Agent to reflect upon a past experience."""
    scenario: str = _get_scenario(log_str)
    query: str = f"""You will be given the history of a past experience in which you were placed in an environment and given a task to complete. You were unsuccessful in completing the task. Do not summarize your environment, but rather think about the strategy and path you took to attempt to complete the task. Devise a concise, new plan of action that accounts for your mistake with reference to specific actions that you should have taken. For example, if you tried A and B but forgot C, then devise a plan to achieve C with environment-specific actions. You will need this later when you are solving the same task. Give your plan after "Plan". Here are two examples:

{FEW_SHOT_EXAMPLES}

{scenario}"""

    if len(memory) > 0:
        query += '\n\nPlans from past attempts:\n'
        for i, m in enumerate(memory):
            query += f'Trial #{i}: {m}\n'

    query += '\n\nNew plan:'
    return query


def update_memory(trial_log_path: str, env_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Updates the given env_config with the appropriate reflections."""
    with open(trial_log_path, 'r') as f:
        full_log: str = f.read()

    env_logs: List[str] = full_log.split('#####\n\n#####')
    assert len(env_logs) == len(env_configs), print(f'bad: {len(env_logs)}, {len(env_configs)}')
    for i, env in enumerate(env_configs):
        # if unsolved, get reflection and update env config
        if not env['is_success'] and not env['skip']:
            if len(env['memory']) > 3:
                memory: List[str] = env['memory'][-3:]
            else:
                memory: List[str] = env['memory']
            reflection_query: str = _generate_reflection_query(env_logs[i], memory)
            reflection: str = get_completion(reflection_query)  # type: ignore
            env_configs[i]['memory'] += [reflection]

    return env_configs


def _generate_counterfactual_query(log_str: str, memory: List[str], v: str) -> str:
    """Allows the Agent to reflect upon a past experience.
    3. Ensure the two items to be placed have distinct numbers.
4. You can only take one item at a time, put it down and pick up the next one later.
5. Do not have more than three counterfactual reasoning analysis or associative reasoning analysis"""
    scenario: str = _get_scenario(log_str)
    query: str = """
You will be given the history of a past experience in which you were placed in an environment and asked to complete a task using the actions in the action list below. You were unsuccessful in completing the task.Therefore, you're going to use causal reasoning to analyze the reasons for your failure so that you can successfully complete the task next time. Follow the examples to generate counterfactual reasoning analysis, associative reasoning analysis and plans.

I will give you some examples to help you better understand how to use causal reasoning to generate counterfactual reasoning analysis, associative reasoning analysis and plan.

Note: 
1. counterfactual reasoning analysis: The following action list is a list of key actions that you may use in completing the task. Review past experiences, check the key actions used in them for grammatical and functional compliance, identify inconsistencies, and use counterfactual reasoning to replace or add the correct action to find the root cause of the error and suggest improvements.
2. Associative reasoning analysis is information about where items are located based on observations in the environment.
3. Ensure the two items to be placed have distinct numbers. 
4. You can only take one item at a time, put it down and pick up the next one later.
5. Do not have more than three counterfactual reasoning analysis or associative reasoning analysis

=================Action List=====================
1.go to {recep}: Instructs the agent to move to the specified container (recep) before it can interact with it.
2.clean {obj} with {recep}: Cleans the specified object (obj) with the specified container (recep).
3.heat {obj} with {recep}: Heat the specified object (obj) with the specified container (recep).
4.put {obj} in/on {recep}: Put the specified object (obj) into the specified container (recep).
5.cool {obj} with {recep}: Cool the specified object (obj) with the specified container (recep).

    """
    query += f"""
=================The first example=====================
{demo[f'{v}_0']}

=================The second example=====================
{demo[f'{v}_1']}

Here is the history you need for Causal reasoning:
{scenario}"""

    if len(memory) > 0:
        query += '\n\nPlans from past attempts:\n'
        for i, m in enumerate(memory):
            query += f'Trial #{i}: {m}\n'

    query += '\n\nCausal reasoning Results:'
    return query


def del_think(scenario):
    line = scenario.split("\n")
    flag = False
    newScenario = ""
    for i, it in enumerate(line):
        if it.startswith("> think:") or it.startswith("OK"):
            if flag:
                continue
            else:
                newScenario += it
                newScenario += '\n'
                flag = True
        else:
            newScenario += it
            newScenario += '\n'
    return newScenario


def _generate_counter_query(log_str: str, memory: List[str], v: str) -> str:
    """Allows the Agent to reflect upon a past experience.

Here are two examples:

=================The first example=====================
{counterfactual[f'{v}_0']}

=================The second example=====================
{counterfactual[f'{v}_1']}


thereby finding the root cause of the task failure. Finally, use counterfactual reasoning to modify or add correct actions based on the syntax requirements in the action list to address the root cause of the failure, thereby solving the fundamental issue. The generated results should focus on different aspects as much as possible.


1. The first root cause of the task failure was due to not being able to heat the {obj} with the microwave. If I add the action "go to microwave 1" before the action "heat {obj} with microwave", then I don't get this failure.
2. The second root cause of the task failure was due to the {obj} was not clean before being put in the {recep}. If I add the action "clean {obj} with sinkbasin 1" after the action "go to sinkbasin 1", then I don't get this failure.
"""
    FOLDER = './prompts'
    PROMPT_FILE = 'counterfactual.json'
    with open(os.path.join(FOLDER, PROMPT_FILE), 'r') as f:
        counterfactual = json.load(f)

    scenario: str = _get_scenario(log_str)
    # scenario = del_think(scenario)

    query: str = """
You will be given the history of a past experience in which you were placed in an environment and given a task to complete. you were unsuccessful in completing the task. You will also be given a list of key actions that can be used to complete the task. Then, review the past experience and identify the key actions that led to task failure or execution failure based on the descriptions in the list of actions, and use counterfactual reasoning to modify or add the correct actions based on the grammatical requirements in the list of actions to get to the root of the problem. Finally, find the root cause of task failure based on the grammatical requirements in the action list. Note: Do not seek the causes of failure from task goal definition or actions in the action list. Ensure tasks can be completed within the range of actions listed. A specific action fails because its prerequisites mentioned in the action list were not met. Do not have more than three counterfactual reasoning analysis. 

=================Action List=====================
When you do not take the object or not go to the container but want to use the following actions, you will fail.
1.heat {obj} with microwave: Heat the specified object (obj) with the microwave.
2.clean {obj} with sinkbasin: Cleans the specified object (obj) with the sinkbasin.
3.cool {obj} with fridge : Cool the specified object (obj) with the fridge.

When you do not see the object or go to the container but want to use the following actions, you will fail.
4.take {obj} from {recep}: Take the specified object (obj) from the specified container (recep).
5.put {obj} in/on {recep}: Put the specified object (obj) into the specified container (recep).

When the target container is not present but you want to use the following actions, you will fail.
6.go to {recep}: Use it when you want to go to the specified container (recep).

When you do not see the container but want to use the following actions, you will fail.
7.open {recep}: Use it when you want to see the items inside the container."""
    query += """
I will give you the example to help you better understand how to use counterfactual reasoning to generate counterfactual reasoning analysis.

=================The example=====================
Counterfactual reasoning analysis:
1. If I add the action "***" before the action "***", then I don't get this failure. The first root cause of the task failure was due to ***.
2. If I add the action "***" after the action "***", then I don't get this failure. The second root cause of the task failure was due to ***.
3. If I correct the action "***" to the action "***", then I don't get this failure. The third root cause of the task failure was due to ***.
    """
#     query += f"""
# I will give you some examples to help you better understand how to use counterfactual reasoning to generate counterfactual reasoning analysis.
#
# =================The first example=====================
# Interact with a household to solve a task. Here is an example.
# You are in the middle of a room. Looking quickly around you, you see a armchair 2, a armchair 1, a bed 1, a countertop 1, a diningtable 1, a drawer 2, a drawer 1, a dresser 1, a garbagecan 1, a laundryhamper 1, and a sidetable 1.
# Your task is to: put two creditcard in dresser.
# > think: To solve the task, I need to find and take the first creditcard, then put it in dresser, then find and take the second creditcard, then put it in dresser.
# OK.
# > think: First I need to find the first creditcard. A creditcard is more likely to appear in drawer (1-2), coutertop (1), sidetable (1), diningtable (1), armchair (1-2), bed (1). I can check one by one, starting with drawer 1.
# OK.
# > go to drawer 1
# The drawer 1 is closed.
# > open drawer 1
# You open the drawer 1. The drawer 1 is open. In it, you see a book 1, a cd 1, and a pen 1.
# > go to drawer 2
# The drawer 2 is closed.
# > open drawer 2
# You open the drawer 2. The drawer 2 is open. In it, you see nothing.
# > go to countertop 1
# On the countertop 1, you see a cellphone 2, a creditcard 4, a creditcard 3, a creditcard 2, a mirror 1, a pencil 2, and a pencil 1.
# > think: Now I find the first creditcard (2). Next, I need to take it. I can find the second creditcard (3) later in countertop 1.
# OK.
# > take creditcard 2 from countertop 1
# You pick up the creditcard 2 from the countertop 1.
# > think: Now I take the first creditcard (2). Next, I need to put it in/on dresser 1.
# OK.
# > go to dresser 1
# On the dresser 1, you see a mug 1, and a television 1.
# > put creditcard 2 in/on dresser 1
# You put the creditcard 2 in/on the dresser 1.
# > think: Now I put the first creditcard in dresser. Next, I need to find the second creditcard. I can directly go to countertop 1.
# OK.
# > go to countertop 1
# On the countertop 1, you see a cellphone 2, a creditcard 4, a creditcard 3, a mirror 1, a pencil 2, and a pencil 1.
# > think: Now I find the second creditcard (3). Next, I need to take it.
# OK.
# > take creditcard 3 from countertop 1
# You pick up the creditcard 3 from the countertop 1.
# > think: Now I take the second creditcard (3). Next, I need to put it in/on dresser 1.
# OK.
# > go to dresser 1
# On the dresser 1, you see a creditcard 2, a mug 1, and a television 1.
# > put creditcard 3 in/on dresser 1
# You put the creditcard 3 in/on the dresser 1.
#
# Here is the task:
# You are in the middle of a room. Looking quickly around you, you see a armchair 1, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a drawer 5, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a dresser 1, a garbagecan 1, a safe 1, a shelf 12, a shelf 11, a shelf 10, a shelf 9, a shelf 8, a shelf 7, a shelf 6, a shelf 5, a shelf 4, a shelf 3, a shelf 2, a shelf 1, a sidetable 1, and a sofa 1.
# Your task is to: put two pillow in sofa.
# > go to armchair 1
# On the armchair 1, you see a cellphone 2, a pillow 2, and a pillow 1.
# > take pillow 1 from armchair 1
# You pick up the pillow 1 from the armchair 1.
# > go to sofa 1
# On the sofa 1, you see a newspaper 3.
# > put pillow 1 in/on sofa 1
# You put the pillow 1 in/on the sofa 1.
# > go to shelf 1
# On the shelf 1, you see nothing.
# > go to sidetable 1
# On the sidetable 1, you see a remotecontrol 1, and a watch 1.
# > go to dresser 1
# On the dresser 1, you see a box 2, a cellphone 1, a creditcard 1, a laptop 1, a remotecontrol 3, a remotecontrol 2, and a statue 1.
# > go to shelf 2
# On the shelf 2, you see nothing.
# > go to sofa 1
# On the sofa 1, you see a newspaper 3, and a pillow 1.
# > take pillow 1 from sofa 1
# You pick up the pillow 1 from the sofa 1.
#
# STATUS: FAIL
#
# Counterfactual reasoning analysis:
# 1.Inadequate memory and information integration: If I remember finding two pillows in an armchair, then I will go towards the armchair to get the second pillow after placing the first one on the sofa, rather than wasting time looking for the second pillow elsewhere.
#     """
    query += f"""

Here is the history you need for counterfactual reasoning:
Interact with a household to solve a task. Here is an examples.
{d[f'react_{v}_1']}

Here is the task:
{scenario}"""

    query += '\n\nCounterfactual reasoning analysis:'
    query = del_think(query)
    return query

def get_association(scenario):
    line = scenario.split("\n")
    newScenario = ""
    for i, it in enumerate(line):
        if it.startswith(">") or it.startswith("OK") or it.startswith('Nothing'):
            continue
        else:
            newScenario += it
            newScenario += '\n'
    return newScenario

def _generate_association_query(log_str: str, memory: List[str], v: str) -> str:
    """Allows the Agent to reflect upon a past experience.
    I will give you some examples of associative reasoning to help you better understand how to generate associative information.
    I will give you some examples of associative reasoning to help you better understand how to generate associative reasoning analysis.

 Based on the environmental information from past experiences, provide the location where you first saw the item related to the task goal.
Note:
1. Using numbers to distinguish different objects.
When no items related to the task goal are found, provide the containers that have been explored , i.e. in which container the search failed.

"""
    FOLDER = './prompts'
    PROMPT_FILE = 'association.json'
    with open(os.path.join(FOLDER, PROMPT_FILE), 'r') as f:
        association = json.load(f)

    scenario: str = _get_scenario(log_str)
    # scenario = get_association(scenario)
    query: str = f"""
You will be given the history of a past experience in which you were placed in an environment. First, identify the names of the items included in the task goal, excluding information about states such as 'clean', 'heat', 'cool', etc. Then, based on the identified item name, review the results of your sighting and determine in which container the item that matches the item name was placed prior to the move, i.e., where you saw the item. When no items related to the task goal are found, provide the containers that have been explored , i.e. in which container the search failed. You do not need to focus on the environmental observations in the examples. Your output only needs to contain what you saw.

Here are two examples:
=================The first example=====================
{association[f'{v}_0']}

=================The second example=====================
{association[f'{v}_1']}

Here is the history of the past experience:
{scenario}"""

    query += '\n\nAssociative reasoning analysis:'
    query = get_association(query)
    return query


def _generate_plan_query(log_str: str, memory: List[str], v: str, counterfactual, association) -> str:
    """Allows the Agent to reflect upon a past experience."""
    FOLDER = './prompts'
    PROMPT_FILE = 'plan.json'
    with open(os.path.join(FOLDER, PROMPT_FILE), 'r') as f:
        plan = json.load(f)

    scenario: str = _get_scenario(log_str)
    scenario = del_think(scenario)
    query: str = f"""
You will be given the history of a past experience in which you were placed in an environment and given a task to complete. You were unsuccessful in completing the task.You will be provided with an analysis of counterfactual reasoning and association reasoning based on your previous experience. Please create a plan for the next attempt, incorporating insights from these two analyses. You need to output in the format of the example without any additional redundant content.
I will give you some examples to help you better understand how to generate plan.

=================The first example=====================
You are in the middle of a room. Looking quickly around you, you see a armchair 1, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a drawer 5, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a dresser 1, a garbagecan 1, a safe 1, a shelf 12, a shelf 11, a shelf 10, a shelf 9, a shelf 8, a shelf 7, a shelf 6, a shelf 5, a shelf 4, a shelf 3, a shelf 2, a shelf 1, a sidetable 1, and a sofa 1.
Your task is to: put two pillow in sofa.
> go to armchair 1
On the armchair 1, you see a cellphone 2, a pillow 2, and a pillow 1.
> take pillow 1 from armchair 1
You pick up the pillow 1 from the armchair 1.
> go to sofa 1
On the sofa 1, you see a newspaper 3.
> put pillow 1 in/on sofa 1
You put the pillow 1 in/on the sofa 1.
> go to shelf 1
On the shelf 1, you see nothing.
> go to sidetable 1
On the sidetable 1, you see a remotecontrol 1, and a watch 1.
> go to dresser 1
On the dresser 1, you see a box 2, a cellphone 1, a creditcard 1, a laptop 1, a remotecontrol 3, a remotecontrol 2, and a statue 1.
> go to shelf 2
On the shelf 2, you see nothing.
> go to sofa 1
On the sofa 1, you see a newspaper 3, and a pillow 1.
> take pillow 1 from sofa 1
You pick up the pillow 1 from the sofa 1.

STATUS: FAIL

Counterfactual reasoning analysis:
1. If I add the action "go to armchair 1" after the action "put pillow 1 in/on sofa 1", then I don't get this failure. The first root cause of the task failure was due to I do not remember finding two pillows in an armchair.

Associative reasoning analysis:
Task goal items: pillow
1. I used to see pillow 1 on armchair 1.
2. I used to see pillow 2 on armchair 1.

Plan:
1.Finding the first pillow: Based on associative reasoning analysis, go to armchair 1 and take pillow 1 from armchair 1.
2.Putting the pillow in the sofa: go to sofa 1 and put pillow 1 in/on sofa 1.
3.Finding the second pillow: Based on associative reasoning analysis, go to armchair 1 and take pillow 2 from armchair 1.
4.Putting the pillows in the sofa: go to sofa 1 and put pillow 2 in/on sofa 1.

Here is the history you need for generating plan:
{scenario}

{counterfactual}

{association}
"""

    query += '\n\nPlan:'
    return query


def update_counterfactual_imaginary(trial_log_path: str, env_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Updates the given env_config with the appropriate reflections."""
    with open(trial_log_path, 'r') as f:
        full_log: str = f.read()

    env_logs: List[str] = full_log.split('#####\n\n#####')
    assert len(env_logs) == len(env_configs), print(f'bad: {len(env_logs)}, {len(env_configs)}')
    for i, env in enumerate(env_configs):
        # if unsolved, get reflection and update env config
        if not env['is_success'] and not env['skip']:
            if len(env['memory']) > 3:
                memory: List[str] = env['memory'][-3:]
            else:
                memory: List[str] = env['memory']

            counterfactual_query = _generate_counterfactual_query(env_logs[i], memory)
            counterfactual: str = get_completion(counterfactual_query)

            print("##########\n")
            print(counterfactual)
            env_configs[i]['memory'] += [counterfactual]

    return env_configs


def qwen_update_counterfactual_imaginary(trial_log_path: str, env_configs: List[Dict[str, Any]]) -> List[
    Dict[str, Any]]:
    """Updates the given env_config with the appropriate reflections."""
    with open(trial_log_path, 'r') as f:
        full_log: str = f.read()

    env_logs: List[str] = full_log.split('#####\n\n#####')
    assert len(env_logs) == len(env_configs), print(f'bad: {len(env_logs)}, {len(env_configs)}')
    for i, env in enumerate(env_configs):
        # if unsolved, get reflection and update env config
        if not env['is_success'] and not env['skip']:
            if len(env['memory']) > 3:
                memory: List[str] = env['memory'][-3:]
            else:
                memory: List[str] = env['memory']

            counterfactual_query = _generate_counterfactual_query(env_logs[i], memory)
            counterfactual: str = qwen_completion(counterfactual_query)

            print("##########\n")
            print(counterfactual)
            env_configs[i]['memory'] += [counterfactual]

    return env_configs


def qwen_update_memory(trial_log_path: str, env_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Updates the given env_config with the appropriate reflections."""
    with open(trial_log_path, 'r') as f:
        full_log: str = f.read()

    env_logs: List[str] = full_log.split('#####\n\n#####')
    assert len(env_logs) == len(env_configs), print(f'bad: {len(env_logs)}, {len(env_configs)}')
    for i, env in enumerate(env_configs):
        # if unsolved, get reflection and update env config
        if not env['is_success'] and not env['skip']:
            if len(env['memory']) > 3:
                memory: List[str] = env['memory'][-3:]
            else:
                memory: List[str] = env['memory']
            reflection_query: str = _generate_reflection_query(env_logs[i], memory)
            reflection: str = qwen_completion(reflection_query)  # type: ignore
            env_configs[i]['memory'] += [reflection]

    return env_configs


def router_update_memory(trial_log_path: str, env_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Updates the given env_config with the appropriate reflections."""
    with open(trial_log_path, 'r') as f:
        full_log: str = f.read()

    env_logs: List[str] = full_log.split('#####\n\n#####')
    assert len(env_logs) == len(env_configs), print(f'bad: {len(env_logs)}, {len(env_configs)}')
    for i, env in enumerate(env_configs):
        # if unsolved, get reflection and update env config
        if not env['is_success'] and not env['skip']:
            if len(env['memory']) > 3:
                memory: List[str] = env['memory'][-3:]
            else:
                memory: List[str] = env['memory']
            reflection_query: str = _generate_reflection_query(env_logs[i], memory)
            reflection: str = router_completion(reflection_query)  # type: ignore
            env_configs[i]['memory'] += [reflection]

    return env_configs


def router_update_counterfactual_imaginary(trial_log_path: str, env_configs: List[Dict[str, Any]]) -> List[
    Dict[str, Any]]:
    """Updates the given env_config with the appropriate reflections."""
    with open(trial_log_path, 'r') as f:
        full_log: str = f.read()

    env_logs: List[str] = full_log.split('#####\n\n#####')
    assert len(env_logs) == len(env_configs), print(f'bad: {len(env_logs)}, {len(env_configs)}')
    for i, env in enumerate(env_configs):
        # if unsolved, get reflection and update env config
        if not env['is_success'] and not env['skip']:
            if len(env['memory']) > 3:
                memory: List[str] = env['memory'][-3:]
            else:
                memory: List[str] = env['memory']

            counter_query = _generate_counter_query(env_logs[i], memory, env['v'])
            counterfactual = router_completion(counter_query)
            if not counterfactual.startswith('Counterfactual reasoning analysis'):
                counterfactual = "Counterfactual reasoning analysis:" + counterfactual

            association_query = _generate_association_query(env_logs[i], memory, env['v'])
            association = router_completion(association_query)
            if not association.startswith('Associative Reasoning Analysis'):
                association = "Associative reasoning analysis:\n" + association
            # # #
            plan_query = _generate_plan_query(env_logs[i], memory, env['v'], counterfactual, association)
            plan = router_completion(plan_query)
            if not plan.startswith('Plan'):
                plan = "Plan:" + plan

            causalinference: str = counterfactual + '\n\n' + association + '\n\n' + plan

            # causalinference_query = _generate_counterfactual_query(env_logs[i], memory, env['v'])
            # causalinference = router_completion(causalinference_query)
            print("##########\n")
            print(causalinference)
            env_configs[i]['memory'] += [causalinference]
    return env_configs
