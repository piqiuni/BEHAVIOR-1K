```
conda activate behavior

python -m omnigibson.examples.environments.behavior_env_demo
python -m omnigibson.examples.environments.behavior_env_ros


python -m omnigibson.examples.robots.robot_control_example

rviz -d ~/Projects/BEHAVIOR-1K/pi_code/pi_rviz.rviz

```

# Eval

```
python OmniGibson/omnigibson/learning/eval.py policy=websocket log_path=$LOG_PATH task.name=$TASK_NAME env_wrapper._target_=$WRAPPER_MODULE


python OmniGibson/omnigibson/learning/eval.py policy=local log_path="/home/rancho/Projects/BEHAVIOR-1K/pi_code/log" task.name="turning_on_radio" env_wrapper._target_="omnigibson.learning.wrappers.RichObservationWrapper"  

```

# 文件位置

```
/home/deepcyber-mk/BEHAVIOR-1K/OmniGibson/omnigibson/configs/r1pro_behavior.yaml


```





# Task Names

```
TASK_NAMES_TO_INDICES = {
    # B10
    "turning_on_radio": 0,
    "picking_up_trash": 1,
    "putting_away_Halloween_decorations": 2,
    "cleaning_up_plates_and_food": 3,
    "can_meat": 4,
    "setting_mousetraps": 5,
    "hiding_Easter_eggs": 6,
    "picking_up_toys": 7,
    "rearranging_kitchen_furniture": 8,
    "putting_up_Christmas_decorations_inside": 9,
    # B20
    "set_up_a_coffee_station_in_your_kitchen": 10,
    "putting_dishes_away_after_cleaning": 11,
    "preparing_lunch_box": 12,
    "loading_the_car": 13,
    "carrying_in_groceries": 14,
    "bringing_in_wood": 15,
    "moving_boxes_to_storage": 16,
    "bringing_water": 17,
    "tidying_bedroom": 18,
    "outfit_a_basic_toolbox": 19,
    # B30
    "sorting_vegetables": 20,
    "collecting_childrens_toys": 21,
    "putting_shoes_on_rack": 22,
    "boxing_books_up_for_storage": 23,
    "storing_food": 24,
    "clearing_food_from_table_into_fridge": 25,
    "assembling_gift_baskets": 26,
    "sorting_household_items": 27,
    "getting_organized_for_work": 28,
    "clean_up_your_desk": 29,
    # B40
    "setting_the_fire": 30,
    "clean_boxing_gloves": 31,
    "wash_a_baseball_cap": 32,
    "wash_dog_toys": 33,
    "hanging_pictures": 34,
    "attach_a_camera_to_a_tripod": 35,
    "clean_a_patio": 36,
    "clean_a_trumpet": 37,
    "spraying_for_bugs": 38,
    "spraying_fruit_trees": 39,
    # B50
    "make_microwave_popcorn": 40,
    "cook_cabbage": 41,
    "chop_an_onion": 42,
    "slicing_vegetables": 43,
    "chopping_wood": 44,
    "cook_hot_dogs": 45,
    "cook_bacon": 46,
    "freeze_pies": 47,
    "canning_food": 48,
    "make_pizza": 49,
}
```
