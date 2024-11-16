def main():
    print("Please select your language:")
    print("1. English")
    print("2. Chinese")

    while True:
        choice = input("Enter your choice (1 or 2): ")
        if choice == '1':
            print("You selected English.")
            return "English"
        elif choice == '2':
            print("You selected Chinese.")
            return "Chinese"
        else:
            print("Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    main()