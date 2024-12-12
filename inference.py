from huggingface_hub import login
#from kaggle_secrets import UserSecretsClient
#user_secrets = UserSecretsClient()

from secret_tokens import hf_token
login(token = hf_token)

prompt = "(CNN) -- Police and FBI agents are investigating the discovery of an empty rocket launcher tube on the front lawn of a Jersey City, New Jersey, home, FBI spokesman Sean Quinn said. Niranjan Desai discovered the 20-year-old AT4 anti-tank rocket launcher tube, a one-time-use device, lying on her lawn Friday morning, police said. The launcher has been turned over to U.S. Army officials at the 754th Ordnance Company, an explosive ordnance disposal unit, at Fort Monmouth, New Jersey, Army officials said. The launcher 'is no longer operable and not considered to be a hazard to public safety,' police said, adding there was no indication the launcher had been fired recently. Army officials said they could not determine if the launcher had been fired, but indicated they should know once they find out where it came from. The nearest military base, Fort Dix, is more than 70 miles from Jersey City. The Joint Terrorism Task Force division of the FBI and Jersey City police are investigating the origin of the rocket launcher and the circumstance that led to its appearance on residential property. 'Al Qaeda doesn't leave a rocket launcher on the lawn of middle-aged ladies,' said Paul Cruickshank of New York University Law School's Center on Law and Security. A neighbor, Joe Quinn, said the object lying on Desai's lawn looked military, was brown, had a handle and strap, and 'both ends were open, like you could shoot something with it.' Quinn also said the device had a picture of a soldier on it and was 3 to 4 feet long. An Army official said the device is basically a shoulder-fired, direct-fire weapon used against ground targets -- a modern-day bazooka -- and it is not wire-guided. According to the Web site Globalsecurity.org, a loaded M136 AT4 anti-tank weapon has a 40-inch-long fiberglass-wrapped tube and weighs just 4 pounds. Its 84 millimeter shaped-charge missile can penetrate 14 inches of armor from a maximum of 985 feet. It is used once and discarded. E-mail to a friend . CNN's Carol Cratty, Dugald McConnell, and Mike Mount contributed to this report."

from transformers import pipeline
import torch

model_id = "google/gemma-2-9b-it"

pipe = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)


messages = [
    {"role": "user", "content": "Generate a short summary for the following text\n" + prompt},
]


outputs = pipe(
    messages,
    max_new_tokens=256,
    do_sample=False,
)


assistant_response = outputs[0]["generated_text"][-1]["content"]
print(assistant_response)

