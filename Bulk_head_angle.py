import numpy as np
import pandas as pd
import os

# Define the input and output directories
input_dir = '/media/tarislada/SAMSUNG/Doric임시/854_0304/representative/postprocessing/savgol_pose'
output_dir = '/media/tarislada/SAMSUNG/Doric임시/854_0304/representative/postprocessing/head_angle'
# input_dir = 'SKH FP/savgol_pose_w59p7'
# output_dir = 'SKH FP/head_angle_w59p7'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Set frame rate for time conversion
frame_rate = 30  # frames per second

# Loop through each CSV file in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.csv'):
        file_path = os.path.join(input_dir, filename)
        
        # Load the CSV file
        column_names = [
            'frame', 'id', 'box x', 'box y', 'box width', 'box height', 'box confidence',
            'nose x', 'nose y',
            'left ear x', 'left ear y', 'right ear x', 'right ear y',
            'left forelimb x', 'left forelimb y', 'right forelimb x', 'right forelimb y',
            'left hindlimb x', 'left hindlimb y', 'right hindlimb x', 'right hindlimb y',
            'tail base x', 'tail base y', 'tail mid x', 'tail mid y', 'tail end x', 'tail end y',
            'body center x', 'body center y',
            'nose conf', 'left ear conf', 'right ear conf', 'left forelimb conf', 'right forelimb conf',
            'left hindlimb conf', 'right hindlimb conf', 'tail root conf', 'tail mid conf', 'tail base conf', 'body center conf'
        ]
        target_columns = ['frame', 'box confidence', 'nose x', 'nose y', 'body center x', 'body center y', 'tail base x', 'tail base y']
        df = pd.read_csv(file_path, header=None, names=column_names)
        
        # Fill missing lines - #TODO: this shit is not right.
        complete_range = pd.DataFrame({'frame': range(int(df['frame'].min()), int(df['frame'].max()) + 1)})
        df_filled = pd.merge(complete_range, df, on='frame', how='left').ffill()
        df = df_filled

        # Sort and keep the row with the highest box confidence for each frame
        df.sort_values(by=['frame', 'box confidence'], ascending=[True, False], inplace=True)
        df.drop_duplicates(subset='frame', keep='first', inplace=True)

        # Select only the target columns
        df_filtered = df[target_columns]

        # Calculate head and body vectors
        df_filtered['head x'] = df_filtered['nose x'] - df_filtered['body center x']
        df_filtered['head y'] = df_filtered['nose y'] - df_filtered['body center y']
        df_filtered['body x'] = df_filtered['body center x'] - df_filtered['tail base x']
        df_filtered['body y'] = df_filtered['body center y'] - df_filtered['tail base y']

        # Create vectors
        head_vector = df_filtered[['head x', 'head y']].values
        body_vector = df_filtered[['body x', 'body y']].values

        # Calculate dot product, cross product, and magnitudes
        dot_product = np.sum(head_vector * body_vector, axis=1)
        cross_product = np.cross(head_vector, body_vector)
        magnitude_head = np.linalg.norm(head_vector, axis=1)
        magnitude_body = np.linalg.norm(body_vector, axis=1)

        # Calculate cosine angle and convert to degrees
        cosine_angle = dot_product / (magnitude_head * magnitude_body)
        angle_radians = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        angle_degrees = np.rad2deg(angle_radians)

        # Adjust angle sign based on cross product
        for i in range(len(angle_degrees)):
            if cross_product[i] > 0:
                angle_degrees[i] = -angle_degrees[i]

        # Add head angle and time columns
        df_filtered['head_angle'] = angle_degrees
        df_filtered['Time'] = df_filtered['frame'] / frame_rate

        # Calculate angular velocity
        df_filtered['angular_velocity'] = df_filtered['head_angle'].diff() / (1 / frame_rate)

        # Keep final columns, including body center coordinates
        final_df = df_filtered[['frame','Time', 'head_angle', 'angular_velocity', 'body center x', 'body center y','nose x', 'nose y', 'tail base x', 'tail base y']]

        # Save the final output to the output directory
        output_path = os.path.join(output_dir, f'processed_{filename}')
        final_df.to_csv(output_path, index=False)
        print(f"Processed file saved as '{output_path}'")

print("All files in the directory have been processed.")
